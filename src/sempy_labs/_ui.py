import ast
import base64
import builtins
import html as html_module
import json
import re
import pandas as pd
import sempy.fabric as fabric
from os import PathLike
from typing import Any, List, Optional
from uuid import UUID, uuid4
from IPython.display import HTML, display
from sempy._utils._log import log
from sempy_labs._helper_functions import _base_api, _create_dataframe, resolve_item_id, resolve_workspace_id, resolve_workspace_name_and_id
from sempy_labs.tom import connect_semantic_model
from autoimport import fix_code

_ignored_notebook_globals = {
    "display",
    "displayHTML",
    "dbutils",
    "mssparkutils",
    "sc",
    "spark",
    "spark_session",
    "sqlContext",
}


def _is_utc_timestamp(value: str) -> bool:
    """Checks whether a string looks like an ISO 8601 UTC timestamp."""

    return value.endswith("Z") and "T" in value


def _decode_b64(payload: str) -> str:
    """Decodes a base64 payload."""

    if not payload:
        return ""

    return base64.b64decode(payload).decode("utf-8", errors="ignore")


def _normalize_value(value: Any) -> str:
    """Converts values to short UI-friendly strings."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _escape_value(value: Any, max_length: int = 180) -> str:
    """Escapes HTML and truncates long values for table cells."""

    text = _normalize_value(value)
    if len(text) > max_length:
        text = f"{text[: max_length - 1]}…"
    return html_module.escape(text)


def _render_cell_value(value: Any, max_length: int = 180) -> str:
    """Renders a table cell, converting UTC timestamps client-side."""

    text = _normalize_value(value)
    display_text = text if len(text) <= max_length else f"{text[: max_length - 1]}…"
    escaped_text = html_module.escape(display_text)

    if isinstance(value, str) and _is_utc_timestamp(value):
        escaped_iso = html_module.escape(value)
        return f'<span data-smu-iso="{escaped_iso}" title="{escaped_iso}">{escaped_text}</span>'

    return escaped_text


def _safe_dataframe_result(title: str, func, formatter=None) -> dict[str, Any]:
    """Executes a dataframe-producing function with per-section error handling."""

    try:
        result = func()
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        if formatter is not None:
            result = formatter(result)
        total_rows = len(result.index)
        preview = result.copy()
        return {
            "kind": "table",
            "title": title,
            "columns": preview.columns.tolist(),
            "rows": preview.fillna("").astype("object").values.tolist(),
            "row_count": int(total_rows),
        }
    except Exception as exc:
        return {
            "kind": "error",
            "title": title,
            "error": str(exc),
        }


def _optional_dataframe_section(title: str, func, formatter=None) -> Optional[dict[str, Any]]:
    """Returns a table section only when there is meaningful content to display."""

    section = _safe_dataframe_result(title=title, func=func, formatter=formatter)
    if section["kind"] == "error":
        error = section["error"]
        if "404 Not Found" in error or "EntityNotFound" in error or "requires the report to be in the PBIR format" in error:
            return None
        return section

    if section["row_count"] == 0:
        return None

    return section


def _select_dataframe_columns(dataframe: pd.DataFrame, columns: list[str], *, exclude_table_rows: bool = False) -> pd.DataFrame:
    """Keeps only the requested columns and optionally excludes table rows."""

    result = dataframe.copy()
    if exclude_table_rows and "Object Type" in result.columns:
        result = result[result["Object Type"] != "Table"]

    available_columns = [column for column in columns if column in result.columns]
    if not available_columns:
        return pd.DataFrame()

    return result.loc[:, available_columns]


def _clean_notebook_python_source(source: str) -> str:
    """Removes notebook magic commands before AST parsing."""

    cleaned_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%%") or stripped.startswith("%") or stripped.startswith("!"):
            cleaned_lines.append("pass")
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _get_notebook_list_request(workspace_id: str | UUID, folder: Optional[str | PathLike | UUID] = None) -> str:
    """Builds the notebook list request."""

    request = f"v1/workspaces/{workspace_id}/notebooks"
    if folder is None:
        return request

    folder_id = fabric.resolve_folder_id(folder=folder, workspace=workspace_id)
    return f"{request}?rootFolderId={folder_id}&recursive=true"


def _get_folder_list_request(workspace_id: str | UUID) -> str:
    """Builds the folder list request."""

    return f"v1/workspaces/{workspace_id}/folders"


def _list_folders(workspace_id: str | UUID) -> pd.DataFrame:
    """Lists workspace folders."""

    columns = {
        "Folder Id": "string",
        "Folder Name": "string",
        "Parent Folder Id": "string",
    }
    df = _create_dataframe(columns=columns)

    try:
        responses = _base_api(
            request=_get_folder_list_request(workspace_id),
            uses_pagination=True,
        )
    except Exception:
        return df

    rows = []
    for response in responses:
        for item in response.get("value", []):
            rows.append(
                {
                    "Folder Id": item.get("id"),
                    "Folder Name": item.get("displayName"),
                    "Parent Folder Id": item.get("parentFolderId"),
                }
            )

    if rows:
        df = pd.DataFrame(rows, columns=list(columns.keys()))

    return df


def _build_folder_paths(df_folders: pd.DataFrame) -> dict[str, str]:
    """Builds folder paths recursively."""

    if df_folders.empty:
        return {}

    folders = {
        str(row["Folder Id"]): {
            "name": row["Folder Name"],
            "parent": row["Parent Folder Id"],
        }
        for _, row in df_folders.iterrows()
        if pd.notna(row["Folder Id"])
    }
    cache = {}

    def resolve_path(folder_id: str) -> str:
        if not folder_id or folder_id not in folders:
            return "/"
        if folder_id in cache:
            return cache[folder_id]

        folder = folders[folder_id]
        name = str(folder["name"])
        parent = folder["parent"]

        if not parent or pd.isna(parent):
            path = f"/{name}"
        else:
            parent_path = resolve_path(str(parent))
            path = f"{parent_path.rstrip('/')}/{name}"

        cache[folder_id] = path
        return path

    return {folder_id: resolve_path(folder_id) for folder_id in folders}


def _extract_import_statements(source: str) -> set[str]:
    """Extracts import statements from source code."""

    try:
        tree = ast.parse(source)
    except Exception:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(f"import {alias.name} as {alias.asname}" if alias.asname else f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.add(f"from {module} import {alias.name} as {alias.asname}" if alias.asname else f"from {module} import {alias.name}")

    return imports


def _suggest_imports_with_autoimport(source: str) -> dict[str, str]:
    """Uses autoimport to suggest missing imports."""

    if fix_code is None:
        return {}

    try:
        fixed_source = fix_code(source)
    except Exception:
        return {}

    original_imports = _extract_import_statements(source)
    fixed_imports = _extract_import_statements(fixed_source)
    added_imports = sorted(fixed_imports - original_imports)
    suggestions = {}

    for import_line in added_imports:
        try:
            tree = ast.parse(import_line)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    exposed_name = alias.asname or alias.name.split(".")[0]
                    suggestions[exposed_name] = import_line
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    exposed_name = alias.asname or alias.name
                    suggestions[exposed_name] = import_line

    return suggestions


def _get_notebook_code_cells(notebook_name: str, workspace: str | UUID) -> list[str]:
    """Gets the python source cells of a Fabric notebook."""

    notebook_id = resolve_item_id(item=notebook_name, type="Notebook", workspace=workspace)
    definition = _base_api(
        request=f"v1/workspaces/{workspace}/notebooks/{notebook_id}/getDefinition",
        method="post",
        client="fabric_sp",
        status_codes=None,
        lro_return_json=True,
    )
    code_cells = []

    for part in definition.get("definition", {}).get("parts", []):
        if part.get("path") != "notebook-content.py":
            continue

        payload = _decode_b64(part.get("payload", ""))
        matches = list(re.finditer(r"(?m)^#.*CELL.*$", payload))

        if not matches:
            code_cells.append(payload)
            continue

        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
            cell_source = payload[start:end].strip("\n")
            if cell_source.strip():
                code_cells.append(cell_source)

    return code_cells


def _load_notebook_sources(workspace: str | UUID, folder: Optional[str | PathLike | UUID] = None) -> pd.DataFrame:
    """Loads notebook metadata and code once."""

    columns = {
        "Notebook Name": "string",
        "Path": "string",
        "Description": "string",
        "Notebook Id": "string",
        "Code Cells": "object",
    }
    df = _create_dataframe(columns=columns)
    workspace_id = resolve_workspace_id(workspace)
    df_notebooks = _list_notebooks_for_ui(workspace=workspace_id, folder=folder)
    rows = []

    for _, notebook_row in df_notebooks.iterrows():
        notebook_name = notebook_row["Notebook Name"]
        notebook_id = notebook_row["Notebook Id"]
        rows.append(
            {
                "Notebook Name": notebook_name,
                "Path": notebook_row["Path"],
                "Description": notebook_row["Description"],
                "Notebook Id": notebook_id,
                "Code Cells": _get_notebook_code_cells(notebook_name=notebook_name, workspace=workspace_id),
            }
        )

    if rows:
        df = pd.DataFrame(rows, columns=list(columns.keys()))

    return df


def _extract_missing_import_issues(sources: list[str], notebook_name: str, notebook_id: str, workspace_name: str, workspace_id: str) -> list[dict[str, str | int]]:
    """Detects names used without import or definition."""

    combined_lines = []
    source_map = {}

    for cell_number, source in enumerate(sources, start=1):
        for cell_line_number, line in enumerate(source.splitlines(), start=1):
            combined_lines.append(line)
            source_map[len(combined_lines)] = (cell_number, cell_line_number, line)
        combined_lines.append("")

    raw_source = "\n".join(combined_lines)
    cleaned_source = _clean_notebook_python_source(raw_source)
    if not cleaned_source.strip():
        return []

    try:
        tree = ast.parse(cleaned_source)
    except SyntaxError as exc:
        cell_number, cell_line_number, line_text = source_map.get(int(exc.lineno or 0), (0, 0, ""))
        return [
            {
                "Workspace Name": workspace_name,
                "Workspace Id": workspace_id,
                "Notebook Name": notebook_name,
                "Notebook Id": notebook_id,
                "Cell Number": cell_number,
                "Line Number": cell_line_number,
                "Issue Type": "Parse Error",
                "Missing Name": "",
                "Suggested Import": "",
                "Line Text": line_text,
                "Details": str(exc),
            }
        ]

    suggested_imports = _suggest_imports_with_autoimport(cleaned_source)
    imported_names = set()
    defined_names = set()
    builtin_names = set(dir(builtins))
    wildcard_import = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    wildcard_import = True
                    continue
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined_names.add(node.id)
        elif isinstance(node, ast.arg):
            defined_names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined_names.add(node.name)

    if wildcard_import:
        return []

    known_names = imported_names | defined_names | builtin_names | _ignored_notebook_globals
    issues = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id in known_names or node.id.startswith("_"):
            continue

        global_line_number = int(node.lineno or 0)
        cell_number, cell_line_number, line_text = source_map.get(global_line_number, (0, 0, ""))
        suggested_import = suggested_imports.get(node.id, "")
        issues.append(
            {
                "Workspace Name": workspace_name,
                "Workspace Id": workspace_id,
                "Notebook Name": notebook_name,
                "Notebook Id": notebook_id,
                "Cell Number": cell_number,
                "Line Number": cell_line_number,
                "Issue Type": "Missing Import",
                "Missing Name": node.id,
                "Suggested Import": suggested_import,
                "Line Text": line_text.strip(),
                "Details": "Import automatically suggested by autoimport." if suggested_import else "Name used without detected import or definition.",
            }
        )

    deduped_issues = []
    seen = set()
    for issue in issues:
        key = (issue["Notebook Id"], issue["Cell Number"], issue["Line Number"], issue["Missing Name"])
        if key in seen:
            continue
        seen.add(key)
        deduped_issues.append(issue)

    return deduped_issues


def _extract_dependencies(sources: list[str], notebook_name: str, notebook_id: str) -> list[dict[str, str | int]]:
    """Extracts notebook dependencies from import statements."""

    rows = []
    for cell_number, source in enumerate(sources, start=1):
        cleaned_source = _clean_notebook_python_source(source)
        try:
            tree = ast.parse(cleaned_source)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    package = alias.name.split(".")[0]
                    rows.append(
                        {
                            "Notebook Name": notebook_name,
                            "Notebook Id": notebook_id,
                            "Cell Number": cell_number,
                            "Import Type": "import",
                            "Package": package,
                            "Import Statement": f"import {alias.name} as {alias.asname}" if alias.asname else f"import {alias.name}",
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                package = module.split(".")[0] if module else ""
                for alias in node.names:
                    rows.append(
                        {
                            "Notebook Name": notebook_name,
                            "Notebook Id": notebook_id,
                            "Cell Number": cell_number,
                            "Import Type": "from",
                            "Package": package,
                            "Import Statement": f"from {module} import {alias.name} as {alias.asname}" if alias.asname else f"from {module} import {alias.name}",
                        }
                    )

    deduped_rows = []
    seen = set()
    for row in rows:
        key = (row["Notebook Id"], row["Import Statement"])
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)

    return deduped_rows


def _extract_magic_commands(sources: list[str], notebook_name: str, notebook_id: str) -> list[dict[str, str | int]]:
    """Extracts notebook magic commands."""

    rows = []
    for cell_number, source in enumerate(sources, start=1):
        for line_number, line in enumerate(source.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("%%"):
                command_type = "Cell Magic"
            elif stripped.startswith("%"):
                command_type = "Line Magic"
            elif stripped.startswith("!"):
                command_type = "Shell Command"
            else:
                continue

            command_name = stripped.split(maxsplit=1)[0]
            rows.append(
                {
                    "Notebook Name": notebook_name,
                    "Notebook Id": notebook_id,
                    "Cell Number": cell_number,
                    "Line Number": line_number,
                    "Command Type": command_type,
                    "Command": command_name,
                    "Line Text": stripped,
                }
            )

    deduped_rows = []
    seen = set()
    for row in rows:
        key = (row["Notebook Id"], row["Cell Number"], row["Line Number"], row["Line Text"])
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)

    return deduped_rows


def _list_notebooks_for_ui(workspace: str | UUID, folder: Optional[str | PathLike | UUID] = None) -> pd.DataFrame:
    """Lists Fabric notebooks with their workspace path."""

    columns = {
        "Notebook Name": "string",
        "Path": "string",
        "Description": "string",
        "Notebook Id": "string",
    }
    df = _create_dataframe(columns=columns)
    workspace_id = resolve_workspace_id(workspace)
    df_folders = _list_folders(workspace_id)
    folder_paths = _build_folder_paths(df_folders)
    responses = _base_api(
        request=_get_notebook_list_request(workspace_id=workspace_id, folder=folder),
        uses_pagination=True,
    )
    rows = []

    for response in responses:
        for item in response.get("value", []):
            folder_id = item.get("folderId")
            path = folder_paths.get(str(folder_id), "/") if folder_id else "/"
            rows.append(
                {
                    "Notebook Name": item.get("displayName"),
                    "Path": path,
                    "Description": item.get("description"),
                    "Notebook Id": item.get("id"),
                }
            )

    if rows:
        df = pd.DataFrame(rows, columns=list(columns.keys()))

    return df


def _search_notebooks_for_ui(search_string: str, workspace: str | UUID | List[str | UUID], notebook: Optional[str | UUID] = None) -> pd.DataFrame:
    """Searches a string inside Fabric notebooks."""

    if isinstance(workspace, (str, UUID)):
        workspace_ids = [resolve_workspace_id(workspace)]
    elif isinstance(workspace, list):
        workspace_ids = [resolve_workspace_id(ws) for ws in workspace]
    else:
        raise ValueError("Workspace must be a string, UUID, or list of strings/UUIDs.")

    df_workspaces = fabric.list_workspaces()
    df_workspaces = df_workspaces[df_workspaces["Id"].isin(workspace_ids)]
    columns = {
        "Workspace Name": "string",
        "Workspace Id": "string",
        "Notebook Name": "string",
        "Notebook Id": "string",
    }
    df = _create_dataframe(columns=columns)
    rows = []

    for _, workspace_row in df_workspaces.iterrows():
        workspace_id = workspace_row["Id"]
        workspace_name = workspace_row["Name"]
        df_notebooks = _list_notebooks_for_ui(workspace=workspace_id)

        if notebook is not None:
            notebook_id = resolve_item_id(item=notebook, type="Notebook", workspace=workspace_id)
            df_notebooks = df_notebooks[df_notebooks["Notebook Id"] == notebook_id]

        for _, notebook_row in df_notebooks.iterrows():
            notebook_id = notebook_row["Notebook Id"]
            notebook_name = notebook_row["Notebook Name"]
            definition = _base_api(
                request=f"v1/workspaces/{workspace_id}/notebooks/{notebook_id}/getDefinition",
                method="post",
                client="fabric_sp",
                status_codes=None,
                lro_return_json=True,
            )

            for part in definition.get("definition", {}).get("parts", []):
                path = part.get("path")
                if path not in ("notebook-content.py", "notebook-content.sql"):
                    continue

                payload = _decode_b64(part.get("payload", ""))
                if search_string in payload:
                    rows.append(
                        {
                            "Workspace Name": workspace_name,
                            "Workspace Id": workspace_id,
                            "Notebook Name": notebook_name,
                            "Notebook Id": notebook_id,
                        }
                    )
                    break

    if rows:
        df = pd.DataFrame(rows, columns=list(columns.keys()))

    return df


def _build_dependencies_summary_from_dependencies(df_dependencies: pd.DataFrame) -> pd.DataFrame:
    """Builds a package-level dependency summary."""

    rows = []
    for _, row in df_dependencies.iterrows():
        packages_text = row.get("Packages", "")
        if not packages_text:
            continue

        packages = [package.strip() for package in str(packages_text).split(",") if package.strip()]
        for package in packages:
            rows.append({"Package": package})

    if not rows:
        return pd.DataFrame(columns=["Package", "Notebook Count"])

    return (
        pd.DataFrame(rows)
        .groupby("Package", as_index=False)
        .size()
        .rename(columns={"size": "Notebook Count"})
        .sort_values("Notebook Count", ascending=False)
        .reset_index(drop=True)
    )


def _build_notebook_snapshot(workspace: str | UUID, folder: Optional[str | PathLike | UUID] = None, search_string: Optional[str] = None) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collects the data used by the notebook UI."""

    workspace_name, workspace_id = resolve_workspace_name_and_id(workspace)
    notebook_sources = _load_notebook_sources(workspace=workspace_id, folder=folder)
    df_notebooks = notebook_sources.drop(columns=["Code Cells"], errors="ignore")
    missing_import_rows = []
    dependency_rows = []
    magic_command_rows = []

    for _, notebook_row in notebook_sources.iterrows():
        notebook_name = notebook_row["Notebook Name"]
        notebook_id = notebook_row["Notebook Id"]
        code_cells = notebook_row["Code Cells"]

        missing_import_rows.extend(
            _extract_missing_import_issues(
                sources=code_cells,
                notebook_name=notebook_name,
                notebook_id=notebook_id,
                workspace_name=workspace_name,
                workspace_id=str(workspace_id),
            )
        )
        dependencies = _extract_dependencies(
            sources=code_cells,
            notebook_name=notebook_name,
            notebook_id=notebook_id,
        )
        magic_command_rows.extend(
            _extract_magic_commands(
                sources=code_cells,
                notebook_name=notebook_name,
                notebook_id=notebook_id,
            )
        )
        packages = sorted({dependency["Package"] for dependency in dependencies if dependency.get("Package")})
        dependency_rows.append(
            {
                "Notebook Name": notebook_name,
                "Package Count": len(packages),
                "Packages": ", ".join(packages),
                "Notebook Id": notebook_id,
            }
        )

    df_missing_imports = (
        pd.DataFrame(missing_import_rows)
        if missing_import_rows
        else pd.DataFrame(
            columns=[
                "Workspace Name",
                "Workspace Id",
                "Notebook Name",
                "Notebook Id",
                "Cell Number",
                "Line Number",
                "Issue Type",
                "Missing Name",
                "Suggested Import",
                "Line Text",
                "Details",
            ]
        )
    )
    if not df_missing_imports.empty:
        df_missing_imports["_has_suggested_import"] = df_missing_imports["Suggested Import"].fillna("").astype(str).ne("")
        df_missing_imports = (
            df_missing_imports
            .sort_values(by=["_has_suggested_import", "Notebook Name", "Missing Name"], ascending=[False, True, True], kind="stable")
            .drop(columns=["_has_suggested_import"])
            .reset_index(drop=True)
        )

    df_dependencies = pd.DataFrame(dependency_rows) if dependency_rows else pd.DataFrame(columns=["Notebook Name", "Package Count", "Packages", "Notebook Id"])
    if not df_dependencies.empty:
        df_dependencies = df_dependencies.sort_values(by=["Notebook Name"], ascending=True, kind="stable").reset_index(drop=True)

    df_magic_commands = (
        pd.DataFrame(magic_command_rows)
        if magic_command_rows
        else pd.DataFrame(
            columns=[
                "Notebook Name",
                "Notebook Id",
                "Cell Number",
                "Line Number",
                "Command Type",
                "Command",
                "Line Text",
            ]
        )
    )
    if not df_magic_commands.empty:
        df_magic_commands = (
            df_magic_commands
            .sort_values(by=["Notebook Name", "Cell Number", "Line Number"], ascending=[True, True, True], kind="stable")
            .reset_index(drop=True)
        )

    df_dependencies_summary = _build_dependencies_summary_from_dependencies(df_dependencies)
    notebooks_with_issues = set(df_missing_imports["Notebook Id"].dropna().astype(str)) if not df_missing_imports.empty else set()
    df_notebooks_ui = df_notebooks.copy()
    if not df_notebooks_ui.empty:
        df_notebooks_ui["_has_issues"] = df_notebooks_ui["Notebook Id"].astype(str).isin(notebooks_with_issues)
        df_notebooks_ui.insert(0, "", df_notebooks_ui["Notebook Id"].astype(str).apply(lambda notebook_id: "❌" if notebook_id in notebooks_with_issues else "✅"))
        df_notebooks_ui = (
            df_notebooks_ui
            .sort_values(by=["_has_issues", "Path", "Notebook Name"], ascending=[False, True, True], kind="stable")
            .drop(columns=["_has_issues"])
            .reset_index(drop=True)
        )

    notebooks_section = _optional_dataframe_section(
        title="Notebooks",
        func=lambda: df_notebooks_ui,
        formatter=lambda dataframe: _select_dataframe_columns(dataframe, ["", "Notebook Name", "Path", "Description", "Notebook Id"]),
    )
    missing_imports_section = _optional_dataframe_section(
        title="Missing Imports",
        func=lambda: df_missing_imports,
        formatter=lambda dataframe: _select_dataframe_columns(dataframe, ["Notebook Name", "Cell Number", "Line Number", "Issue Type", "Missing Name", "Suggested Import"]),
    )
    dependencies_by_notebook_section = _optional_dataframe_section(
        title="Dependencies By Notebook",
        func=lambda: df_dependencies,
        formatter=lambda dataframe: _select_dataframe_columns(dataframe, ["Notebook Name", "Package Count", "Packages", "Notebook Id"]),
    )
    dependencies_summary_section = _optional_dataframe_section(
        title="Dependencies Summary",
        func=lambda: df_dependencies_summary,
        formatter=lambda dataframe: _select_dataframe_columns(dataframe, ["Package", "Notebook Count"]),
    )
    magic_commands_section = _optional_dataframe_section(
        title="Magic Commands",
        func=lambda: df_magic_commands,
        formatter=lambda dataframe: _select_dataframe_columns(dataframe, ["Notebook Name", "Cell Number", "Line Number", "Command Type", "Command", "Line Text"]),
    )

    search_sections = []
    if search_string:
        search_section = _optional_dataframe_section(
            title=f'Search Results - "{search_string}"',
            func=lambda: _search_notebooks_for_ui(search_string=search_string, workspace=workspace_id),
        )
        if search_section is not None:
            search_sections.append(search_section)

    metrics = [
        {"label": "Notebooks", "value": int(len(df_notebooks.index))},
        {"label": "Notebooks KO", "value": int(df_missing_imports["Notebook Id"].nunique()) if not df_missing_imports.empty else 0},
    ]
    tabs = [
        {"id": "notebooks", "label": "Notebooks", "sections": [section for section in [notebooks_section] if section is not None]},
        {"id": "imports", "label": "Missing Imports", "sections": [section for section in [missing_imports_section] if section is not None]},
        {"id": "magic-commands", "label": "Magic Commands", "sections": [section for section in [magic_commands_section] if section is not None]},
        {"id": "dependencies", "label": "Dependencies", "sections": [section for section in [dependencies_by_notebook_section, dependencies_summary_section] if section is not None]},
    ]
    if search_sections:
        tabs.append({"id": "search", "label": "Search", "sections": search_sections})
    tabs = [tab for tab in tabs if tab["sections"]]

    return str(workspace_name), str(workspace_id), metrics, tabs


def _build_semantic_model_snapshot(dataset: str | UUID, workspace: Optional[str | UUID]) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collects the data used by the notebook UI."""

    from sempy_labs._data_access_security import list_data_access_roles
    from sempy_labs._list_functions import get_object_level_security, list_columns, list_relationships, list_reports_using_semantic_model, list_tables
    from sempy_labs._refresh_semantic_model import get_semantic_model_refresh_history
    from sempy_labs._semantic_models import get_semantic_model_refresh_schedule

    with connect_semantic_model(dataset=dataset, workspace=workspace, readonly=True) as tom:
        model_name = str(tom._dataset_name)
        workspace_name = str(tom._workspace_name)

        tables: list[dict[str, Any]] = []
        roles: list[dict[str, Any]] = []
        perspectives: list[dict[str, Any]] = []

        measure_count = 0
        column_count = 0
        hierarchy_count = 0
        calculation_item_count = 0
        relationship_count = tom.model.Relationships.Count
        function_count = tom.model.Functions.Count

        for table in tom.model.Tables:
            columns = [c.Name for c in table.Columns if str(c.Type) != "RowNumber"]
            measures = [m.Name for m in table.Measures]
            hierarchies = [h.Name for h in table.Hierarchies]
            calc_items = [ci.Name for ci in table.CalculationGroup.CalculationItems] if table.CalculationGroup is not None else []
            modes: list[str] = []
            for partition in table.Partitions:
                mode = getattr(partition, "Mode", None)
                if mode is None:
                    continue
                mode_name = str(mode)
                if mode_name not in modes:
                    modes.append(mode_name)

            column_count += len(columns)
            measure_count += len(measures)
            hierarchy_count += len(hierarchies)
            calculation_item_count += len(calc_items)

            tables.append(
                {
                    "name": table.Name,
                    "type": "Calculation Group" if table.CalculationGroup is not None else ("Calculated Table" if tom.is_calculated_table(table_name=table.Name) else "Table"),
                    "columns": columns,
                    "measures": measures,
                    "hierarchies": hierarchies,
                    "calculation_items": calc_items,
                    "modes": modes,
                }
            )
        for perspective in tom.model.Perspectives:
            members = {}
            for table in tom.model.Tables:
                table_columns = [c.Name for c in table.Columns if str(c.Type) != "RowNumber" and tom.in_perspective(c, perspective.Name)]
                table_measures = [m.Name for m in table.Measures if tom.in_perspective(m, perspective.Name)]
                table_hierarchies = [h.Name for h in table.Hierarchies if tom.in_perspective(h, perspective.Name)]
                if table_columns or table_measures or table_hierarchies:
                    members[table.Name] = {
                        "columns": table_columns,
                        "measures": table_measures,
                        "hierarchies": table_hierarchies,
                    }
            perspectives.append({"name": perspective.Name, "members": members})
        for role in tom.model.Roles:
            filters = {}
            for permission in role.TablePermissions:
                expression = permission.FilterExpression
                if expression:
                    filters[permission.Name] = expression
            roles.append({"name": role.Name, "filters": filters})
    metrics = [
        {"label": "Tables", "value": len(tables)},
        {"label": "Columns", "value": column_count},
        {"label": "Measures", "value": measure_count},
        {"label": "Relationships", "value": relationship_count},
    ]

    overview_sections = [
        {
            "kind": "tree",
            "title": "Model Explorer",
            "tables": tables,
        }
    ]
    if perspectives:
        overview_sections.append(
            {
                "kind": "code",
                "title": "Perspectives",
                "language": "json",
                "content": json.dumps(perspectives, indent=2, ensure_ascii=True),
            }
        )
    if roles:
        overview_sections.append(
            {
                "kind": "code",
                "title": "Roles And RLS",
                "language": "json",
                "content": json.dumps(roles, indent=2, ensure_ascii=True),
            }
        )

    structure_sections = [
        _optional_dataframe_section("Tables", lambda: list_tables(dataset=dataset, workspace=workspace, extended=True)),
        _optional_dataframe_section("Columns", lambda: list_columns(dataset=dataset, workspace=workspace)),
        _optional_dataframe_section("Relationships", lambda: list_relationships(dataset=dataset, workspace=workspace, extended=True)),
    ]
    structure_sections = [section for section in structure_sections if section is not None]
    security_sections = [_optional_dataframe_section("Object Level Security", lambda: get_object_level_security(dataset=dataset, workspace=workspace))]
    security_sections = [section for section in security_sections if section is not None]

    data_access_roles_section = _optional_dataframe_section("OneLake Data Access Roles", lambda: list_data_access_roles(item=dataset, type="SemanticModel", workspace=workspace))
    if data_access_roles_section is not None:
        security_sections.append(data_access_roles_section)

    usage_sections = [_optional_dataframe_section("Reports Using Semantic Model", lambda: list_reports_using_semantic_model(dataset=dataset, workspace=workspace))]
    usage_sections = [section for section in usage_sections if section is not None]

    quality_sections = [
        _optional_dataframe_section("Refresh Schedule", lambda: get_semantic_model_refresh_schedule(dataset=dataset, workspace=workspace)),
        _optional_dataframe_section("Refresh History", lambda: get_semantic_model_refresh_history(dataset=dataset, workspace=workspace), formatter=lambda dataframe: _select_dataframe_columns(dataframe, ["Refresh Type", "Start Time", "End Time", "Error Code", "Error Description", "Status", "Extended Status"])),
    ]
    quality_sections = [section for section in quality_sections if section is not None]

    tabs = [
        {"id": "overview", "label": "Overview", "sections": overview_sections},
        {"id": "structure", "label": "Structure", "sections": structure_sections},
        {"id": "security", "label": "Security", "sections": security_sections},
        {"id": "usage", "label": "Usage", "sections": usage_sections},
        {"id": "quality", "label": "Quality", "sections": quality_sections},
    ]
    tabs = [tab for tab in tabs if tab["sections"]]

    return model_name, workspace_name, metrics, tabs, tables


def _render_semantic_model_ui(model_name: str, workspace_name: str, metrics: list[dict[str, Any]], tabs: list[dict[str, Any]]) -> None:
    """Renders the semantic model notebook UI."""

    uid = uuid4().hex[:8]

    styles = f"""
    <style>
    html, body, body > div {{
        margin: 0;
        padding: 0;
        border: 0;
        font-family: "Segoe UI", "Segoe UI Web (West European)", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif;
    }}
    .smu-{uid} {{
        --vpx-accent: #0071e3;
        --vpx-accent-hover: #0077ed;
        --vpx-bg: #ffffff;
        --vpx-bg-secondary: #f5f5f7;
        --vpx-bg-tertiary: #fbfbfd;
        --vpx-border: rgba(0, 0, 0, 0.06);
        --vpx-border-strong: rgba(0, 0, 0, 0.12);
        --vpx-text: #1d1d1f;
        --vpx-text-secondary: #6e6e73;
        --vpx-text-tertiary: #86868b;
        --vpx-shadow-lg: 0 12px 40px rgba(0,0,0,0.12), 0 4px 12px rgba(0,0,0,0.06);
        --vpx-radius: 12px;
        --vpx-radius-sm: 8px;
        --vpx-transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        color: var(--vpx-text);
        -webkit-font-smoothing: antialiased;
        max-width: 100%;
        margin: 0;
        padding: 0;
    }}
    .smu-{uid} *, .smu-{uid} *::before, .smu-{uid} *::after {{
        box-sizing: border-box;
    }}
    .smu-{uid} .smu-shell {{
        background: var(--vpx-bg);
        border-radius: var(--vpx-radius);
        box-shadow: var(--vpx-shadow-lg);
        overflow: hidden;
        border: 1px solid var(--vpx-border);
    }}
    .smu-{uid} .smu-header {{
        padding: 20px 24px 0 24px;
        background: var(--vpx-bg);
    }}
    .smu-{uid} .smu-title {{
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--vpx-text);
        margin: 0 0 4px 0;
        line-height: 1.2;
    }}
    .smu-{uid} .smu-subtitle {{
        font-size: 12px;
        color: var(--vpx-text-tertiary);
        margin: 0 0 16px 0;
    }}
    .smu-{uid} .smu-metrics {{
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 12px;
        padding: 0 24px 16px 24px;
    }}
    .smu-{uid} .smu-metric {{
        background: var(--vpx-bg-secondary);
        border: 1px solid var(--vpx-border);
        border-radius: var(--vpx-radius-sm);
        padding: 14px 14px 12px;
    }}
    .smu-{uid} .smu-metric-value {{
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--vpx-text);
        font-variant-numeric: tabular-nums;
    }}
    .smu-{uid} .smu-metric-label {{
        margin-top: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--vpx-text-tertiary);
    }}
    .smu-{uid} .smu-tabs {{
        display: flex;
        gap: 2px;
        padding: 0 24px;
        overflow-x: auto;
        scrollbar-width: none;
        -ms-overflow-style: none;
        background: var(--vpx-bg);
        border-bottom: 1px solid var(--vpx-border);
    }}
    .smu-{uid} .smu-tabs::-webkit-scrollbar {{ display: none; }}
    .smu-{uid} .smu-tab {{
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: none;
        background: transparent;
        color: var(--vpx-text-secondary);
        font-size: 13px;
        font-weight: 500;
        padding: 10px 16px;
        cursor: pointer;
        white-space: nowrap;
        transition: color var(--vpx-transition);
    }}
    .smu-{uid} .smu-tab.smu-active {{
        color: var(--vpx-accent);
        font-weight: 600;
    }}
    .smu-{uid} .smu-tab::after {{
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--vpx-accent);
        border-radius: 2px 2px 0 0;
        transform: scaleX(0);
        transition: transform var(--vpx-transition);
    }}
    .smu-{uid} .smu-tab.smu-active::after {{
        transform: scaleX(1);
    }}
    .smu-{uid} .smu-panel {{
        display: none;
        padding: 0;
        background: var(--vpx-bg);
    }}
    .smu-{uid} .smu-panel.smu-visible {{
        display: block;
    }}
    .smu-{uid} .smu-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 12px;
        padding: 16px 24px 24px 24px;
    }}
    .smu-{uid} .smu-section {{
        border: 1px solid var(--vpx-border);
        border-radius: var(--vpx-radius-sm);
        background: var(--vpx-bg);
        overflow: hidden;
        min-width: 0;
    }}
    .smu-{uid} .smu-section.smu-span {{
        grid-column: 1 / -1;
    }}
    .smu-{uid} .smu-section-head {{
        padding: 14px 16px;
        border-bottom: 1px solid var(--vpx-border);
        background: var(--vpx-bg-secondary);
    }}
    .smu-{uid} .smu-section-title {{
        margin: 0;
        font-size: 14px;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }}
    .smu-{uid} .smu-section-body {{
        padding: 0;
    }}
    .smu-{uid} .smu-note {{
        font-size: 12px;
        color: var(--vpx-text-tertiary);
    }}
    .smu-{uid} .smu-search {{
        width: 260px;
        padding: 7px 12px 7px 12px;
        font-size: 13px;
        background: var(--vpx-bg);
        border: 1px solid var(--vpx-border-strong);
        border-radius: var(--vpx-radius-sm);
        color: var(--vpx-text);
        outline: none;
    }}
    .smu-{uid} .smu-table-wrap {{
        overflow: auto;
        max-height: 560px;
    }}
    .smu-{uid} table {{
        width: max-content;
        min-width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        line-height: 1.4;
        table-layout: fixed;
    }}
    .smu-{uid} th,
    .smu-{uid} td {{
        padding: 9px 16px;
        border-bottom: 1px solid var(--vpx-border);
        text-align: left;
        vertical-align: top;
        white-space: nowrap;
    }}
    .smu-{uid} th {{
        position: sticky;
        top: 0;
        padding: 10px 16px;
        background: var(--vpx-bg-secondary);
        z-index: 1;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--vpx-text-secondary);
    }}
    .smu-{uid} .smu-code {{
        background: var(--vpx-bg-tertiary);
        border: none;
        padding: 14px;
        overflow: auto;
        max-height: 560px;
        font-family: Consolas, "Courier New", monospace;
        font-size: 13px;
        white-space: pre-wrap;
    }}
    .smu-{uid} .smu-error {{
        border: none;
        color: #323130;
        background: #fde7e9;
        padding: 14px 16px;
        font-size: 12px;
        white-space: pre-wrap;
    }}
    .smu-{uid} .smu-tree {{
        display: block;
    }}
    .smu-{uid} .smu-tree-card {{
        border-bottom: 1px solid var(--vpx-border);
        background: var(--vpx-bg);
    }}
    .smu-{uid} .smu-tree-head {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 11px 16px;
        cursor: pointer;
    }}
    .smu-{uid} .smu-tree-name {{
        font-weight: 600;
        font-size: 13px;
    }}
    .smu-{uid} .smu-tree-type {{
        color: var(--vpx-text-tertiary);
        font-size: 11px;
        margin-top: 3px;
    }}
    .smu-{uid} .smu-tree-badge {{
        align-self: center;
        font-size: 11px;
        color: var(--vpx-text-tertiary);
        background: var(--vpx-bg-secondary);
        padding: 3px 8px;
        border-radius: 999px;
        white-space: nowrap;
    }}
    .smu-{uid} .smu-tree-badges {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
    }}
    .smu-{uid} .smu-mode-badge {{
        align-self: center;
        font-size: 11px;
        color: var(--vpx-accent);
        background: rgba(0, 113, 227, 0.08);
        border: 1px solid rgba(0, 113, 227, 0.16);
        padding: 3px 8px;
        border-radius: 999px;
        white-space: nowrap;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .smu-{uid} .smu-tree-body {{
        display: none;
        padding: 12px 13px 14px;
    }}
    .smu-{uid} .smu-tree-card.smu-open .smu-tree-body {{
        display: block;
    }}
    .smu-{uid} .smu-tree-group + .smu-tree-group {{
        margin-top: 10px;
    }}
    .smu-{uid} .smu-tree-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--vpx-text-tertiary);
        margin-bottom: 6px;
    }}
    .smu-{uid} .smu-chip-list {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }}
    .smu-{uid} .smu-chip {{
        background: var(--vpx-bg-secondary);
        border: 1px solid var(--vpx-border);
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 11px;
    }}
    .smu-{uid} .smu-message {{
        font-size: 12px;
        color: var(--vpx-text-tertiary);
        background: var(--vpx-bg-tertiary);
        padding: 14px 16px;
    }}
    </style>
    """

    html_parts = [f'<div class="smu-{uid}"><div class="smu-shell">']
    html_parts.append('<div class="smu-header">')
    html_parts.append(f'<h1 class="smu-title">Semantic Model Explorer — {html_module.escape(model_name)}</h1>')
    html_parts.append(f'<div class="smu-subtitle">{html_module.escape(workspace_name)}</div>')
    html_parts.append("</div>")

    html_parts.append('<div class="smu-metrics">')
    for metric in metrics:
        html_parts.append('<div class="smu-metric">')
        html_parts.append(f'<div class="smu-metric-value">{html_module.escape(str(metric["value"]))}</div>')
        html_parts.append(f'<div class="smu-metric-label">{html_module.escape(metric["label"])}</div>')
        html_parts.append("</div>")
    html_parts.append("</div>")

    html_parts.append('<div class="smu-tabs">')
    for index, tab in enumerate(tabs):
        active = " smu-active" if index == 0 else ""
        html_parts.append(f'<button class="smu-tab{active}" data-smu-tab="{html_module.escape(tab["id"])}">{html_module.escape(tab["label"])}</button>')
    html_parts.append("</div>")

    for index, tab in enumerate(tabs):
        visible = " smu-visible" if index == 0 else ""
        html_parts.append(f'<div class="smu-panel{visible}" data-smu-panel="{html_module.escape(tab["id"])}">')
        html_parts.append('<div class="smu-grid">')
        for section in tab["sections"]:
            span_class = " smu-span" if section.get("kind") in {"table", "tree", "code"} else ""
            html_parts.append(f'<section class="smu-section{span_class}">')
            html_parts.append('<div class="smu-section-head">')
            html_parts.append(f'<h2 class="smu-section-title">{html_module.escape(section["title"])}</h2>')
            html_parts.append("</div>")
            html_parts.append('<div class="smu-section-body">')

            kind = section["kind"]
            if kind == "table":
                note = f'{section["row_count"]} row(s)'
                search_id = f"smu-search-{uid}-{tab['id']}-{section['title']}".replace(" ", "-")
                html_parts.append(
                    '<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--vpx-border);background:var(--vpx-bg-tertiary)">'
                    f'<input class="smu-search" id="{html_module.escape(search_id)}" placeholder="Filter rows…" />'
                    f'<div class="smu-note" id="{html_module.escape(search_id)}-count">{html_module.escape(note)}</div>'
                    '</div>'
                )
                html_parts.append('<div class="smu-table-wrap">')
                html_parts.append(f'<table data-smu-search-target="{html_module.escape(search_id)}"><thead><tr>')
                for column in section["columns"]:
                    html_parts.append(f"<th>{html_module.escape(str(column))}</th>")
                html_parts.append("</tr></thead><tbody>")
                for row in section["rows"]:
                    html_parts.append("<tr>")
                    for cell in row:
                        html_parts.append(f"<td>{_render_cell_value(cell)}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</tbody></table></div>")
            elif kind == "code":
                html_parts.append(f'<pre class="smu-code">{html_module.escape(section["content"])}</pre>')
            elif kind == "tree":
                html_parts.append('<div class="smu-tree">')
                for table in section["tables"]:
                    object_count = len(table["columns"]) + len(table["measures"]) + len(table["hierarchies"]) + len(table["calculation_items"])
                    html_parts.append('<div class="smu-tree-card">')
                    html_parts.append('<div class="smu-tree-head">')
                    html_parts.append('<div>')
                    html_parts.append(f'<div class="smu-tree-name">{html_module.escape(table["name"])}</div>')
                    html_parts.append(f'<div class="smu-tree-type">{html_module.escape(table["type"])}</div>')
                    html_parts.append('</div>')
                    html_parts.append('<div class="smu-tree-badges">')
                    for mode in table.get("modes", []):
                        html_parts.append(f'<div class="smu-mode-badge">{html_module.escape(mode)}</div>')
                    html_parts.append(f'<div class="smu-tree-badge">{object_count} object(s)</div>')
                    html_parts.append('</div>')
                    html_parts.append('</div>')
                    html_parts.append('<div class="smu-tree-body">')
                    for key, label in [
                        ("columns", "Columns"),
                        ("measures", "Measures"),
                        ("hierarchies", "Hierarchies"),
                        ("calculation_items", "Calculation Items"),
                    ]:
                        if table[key]:
                            html_parts.append('<div class="smu-tree-group">')
                            html_parts.append(f'<div class="smu-tree-label">{html_module.escape(label)}</div>')
                            html_parts.append('<div class="smu-chip-list">')
                            for value in table[key]:
                                html_parts.append(f'<span class="smu-chip">{html_module.escape(value)}</span>')
                            html_parts.append('</div></div>')
                    html_parts.append('</div></div>')
                html_parts.append("</div>")
            elif kind == "message":
                html_parts.append(f'<div class="smu-message">{html_module.escape(section["content"])}</div>')
            else:
                html_parts.append(f'<div class="smu-error">{html_module.escape(section.get("error", "Unknown error"))}</div>')

            html_parts.append("</div></section>")
        html_parts.append("</div></div>")

    html_parts.append('<div style="padding:10px 24px;font-size:11px;color:var(--vpx-text-tertiary);text-align:right;border-top:1px solid var(--vpx-border);background:var(--vpx-bg-tertiary)">made by jocelyn with &#10084;</div>')
    html_parts.append("</div></div>")

    script = f"""
    <script>
    (function() {{
        var root = document.querySelector('.smu-{uid}');
        if (!root) return;
        var tabs = Array.from(root.querySelectorAll('.smu-tab'));
        var panels = Array.from(root.querySelectorAll('.smu-panel'));

        tabs.forEach(function(tab) {{
            tab.addEventListener('click', function() {{
                var target = tab.getAttribute('data-smu-tab');
                tabs.forEach(function(item) {{ item.classList.remove('smu-active'); }});
                panels.forEach(function(panel) {{ panel.classList.remove('smu-visible'); }});
                tab.classList.add('smu-active');
                var panel = root.querySelector('[data-smu-panel="' + target + '"]');
                if (panel) panel.classList.add('smu-visible');
            }});
        }});

        root.querySelectorAll('.smu-search').forEach(function(input) {{
            input.addEventListener('input', function() {{
                var table = root.querySelector('table[data-smu-search-target="' + input.id + '"]');
                if (!table) return;
                var query = input.value.toLowerCase();
                var rows = Array.from(table.querySelectorAll('tbody tr'));
                var shown = 0;
                rows.forEach(function(row) {{
                    var text = row.textContent.toLowerCase();
                    var visible = !query || text.indexOf(query) !== -1;
                    row.style.display = visible ? '' : 'none';
                    if (visible) shown++;
                }});
                var counter = document.getElementById(input.id + '-count');
                if (counter) {{
                    counter.textContent = shown + ' row' + (shown !== 1 ? 's' : '');
                }}
            }});
        }});

        root.querySelectorAll('.smu-tree-head').forEach(function(head) {{
            head.addEventListener('click', function() {{
                var card = head.closest('.smu-tree-card');
                if (card) card.classList.toggle('smu-open');
            }});
        }});

        root.querySelectorAll('[data-smu-iso]').forEach(function(node) {{
            var iso = node.getAttribute('data-smu-iso');
            if (!iso) return;
            var dt = new Date(iso);
            if (Number.isNaN(dt.getTime())) return;
            node.textContent = dt.toLocaleString();
            node.title = iso;
        }});
    }})();
    </script>
    """

    display(HTML(styles + "".join(html_parts) + script))


@log
def explore_semantic_model(dataset: str | UUID, workspace: Optional[str | UUID] = None) -> None:
    """
    Opens a notebook UI for inspecting a semantic model in depth.

    This read-only UI consolidates structure, security, report usage,
    and quality checks into a single notebook-rendered experience.

    Parameters
    ----------
    dataset : str | uuid.UUID
        Name or ID of the semantic model.
    workspace : str | uuid.UUID, default=None
        The Fabric workspace name or ID.
        Defaults to None which resolves to the workspace of the attached lakehouse
        or if no lakehouse attached, resolves to the workspace of the notebook.
    """

    model_name, workspace_name, metrics, tabs, _tables = _build_semantic_model_snapshot(dataset=dataset, workspace=workspace)
    _render_semantic_model_ui(model_name=model_name, workspace_name=workspace_name, metrics=metrics, tabs=tabs)


def _render_notebook_ui(workspace_name: str, workspace_id: str, metrics: list[dict[str, Any]], tabs: list[dict[str, Any]]) -> None:
    """Renders the notebook explorer UI."""

    uid = uuid4().hex[:8]

    styles = f"""
    <style>
    html, body, body > div {{
        margin: 0;
        padding: 0;
        border: 0;
        font-family: "Segoe UI", "Segoe UI Web (West European)", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif;
    }}
    .nbe-{uid} {{
        --vpx-accent: #0071e3;
        --vpx-accent-hover: #0077ed;
        --vpx-bg: #ffffff;
        --vpx-bg-secondary: #f5f5f7;
        --vpx-bg-tertiary: #fbfbfd;
        --vpx-border: rgba(0, 0, 0, 0.06);
        --vpx-border-strong: rgba(0, 0, 0, 0.12);
        --vpx-text: #1d1d1f;
        --vpx-text-secondary: #6e6e73;
        --vpx-text-tertiary: #86868b;
        --vpx-shadow-lg: 0 12px 40px rgba(0,0,0,0.12), 0 4px 12px rgba(0,0,0,0.06);
        --vpx-radius: 12px;
        --vpx-radius-sm: 8px;
        --vpx-transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        color: var(--vpx-text);
        -webkit-font-smoothing: antialiased;
        max-width: 100%;
        margin: 0;
        padding: 0;
    }}
    .nbe-{uid} *, .nbe-{uid} *::before, .nbe-{uid} *::after {{
        box-sizing: border-box;
    }}
    .nbe-{uid} .nbe-shell {{
        background: var(--vpx-bg);
        border-radius: var(--vpx-radius);
        box-shadow: var(--vpx-shadow-lg);
        overflow: hidden;
        border: 1px solid var(--vpx-border);
    }}
    .nbe-{uid} .nbe-header {{
        padding: 20px 24px 0 24px;
        background: var(--vpx-bg);
    }}
    .nbe-{uid} .nbe-title {{
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--vpx-text);
        margin: 0 0 4px 0;
        line-height: 1.2;
    }}
    .nbe-{uid} .nbe-subtitle {{
        font-size: 12px;
        color: var(--vpx-text-tertiary);
        margin: 0 0 16px 0;
    }}
    .nbe-{uid} .nbe-metrics {{
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 12px;
        padding: 0 24px 16px 24px;
    }}
    .nbe-{uid} .nbe-metric {{
        background: var(--vpx-bg-secondary);
        border: 1px solid var(--vpx-border);
        border-radius: var(--vpx-radius-sm);
        padding: 14px 14px 12px;
    }}
    .nbe-{uid} .nbe-metric-value {{
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--vpx-text);
        font-variant-numeric: tabular-nums;
    }}
    .nbe-{uid} .nbe-metric-label {{
        margin-top: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--vpx-text-tertiary);
    }}
    .nbe-{uid} .nbe-tabs {{
        display: flex;
        gap: 2px;
        padding: 0 24px;
        overflow-x: auto;
        scrollbar-width: none;
        -ms-overflow-style: none;
        background: var(--vpx-bg);
        border-bottom: 1px solid var(--vpx-border);
    }}
    .nbe-{uid} .nbe-tabs::-webkit-scrollbar {{ display: none; }}
    .nbe-{uid} .nbe-tab {{
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: none;
        background: transparent;
        color: var(--vpx-text-secondary);
        font-size: 13px;
        font-weight: 500;
        padding: 10px 16px;
        cursor: pointer;
        white-space: nowrap;
        transition: color var(--vpx-transition);
    }}
    .nbe-{uid} .nbe-tab.nbe-active {{
        color: var(--vpx-accent);
        font-weight: 600;
    }}
    .nbe-{uid} .nbe-tab::after {{
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--vpx-accent);
        border-radius: 2px 2px 0 0;
        transform: scaleX(0);
        transition: transform var(--vpx-transition);
    }}
    .nbe-{uid} .nbe-tab.nbe-active::after {{
        transform: scaleX(1);
    }}
    .nbe-{uid} .nbe-panel {{
        display: none;
        padding: 0;
        background: var(--vpx-bg);
    }}
    .nbe-{uid} .nbe-panel.nbe-visible {{
        display: block;
    }}
    .nbe-{uid} .nbe-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 12px;
        padding: 16px 24px 24px 24px;
    }}
    .nbe-{uid} .nbe-section {{
        border: 1px solid var(--vpx-border);
        border-radius: var(--vpx-radius-sm);
        background: var(--vpx-bg);
        overflow: hidden;
        min-width: 0;
    }}
    .nbe-{uid} .nbe-section.nbe-span {{
        grid-column: 1 / -1;
    }}
    .nbe-{uid} .nbe-section-head {{
        padding: 14px 16px;
        border-bottom: 1px solid var(--vpx-border);
        background: var(--vpx-bg-secondary);
    }}
    .nbe-{uid} .nbe-section-title {{
        margin: 0;
        font-size: 14px;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }}
    .nbe-{uid} .nbe-section-body {{
        padding: 0;
    }}
    .nbe-{uid} .nbe-note {{
        font-size: 12px;
        color: var(--vpx-text-tertiary);
    }}
    .nbe-{uid} .nbe-search {{
        width: 260px;
        padding: 7px 12px 7px 12px;
        font-size: 13px;
        background: var(--vpx-bg);
        border: 1px solid var(--vpx-border-strong);
        border-radius: var(--vpx-radius-sm);
        color: var(--vpx-text);
        outline: none;
    }}
    .nbe-{uid} .nbe-table-wrap {{
        overflow: auto;
        max-height: 560px;
    }}
    .nbe-{uid} table {{
        width: max-content;
        min-width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        line-height: 1.4;
        table-layout: fixed;
    }}
    .nbe-{uid} th,
    .nbe-{uid} td {{
        padding: 9px 16px;
        border-bottom: 1px solid var(--vpx-border);
        text-align: left;
        vertical-align: top;
        white-space: nowrap;
    }}
    .nbe-{uid} th {{
        position: sticky;
        top: 0;
        padding: 10px 16px;
        background: var(--vpx-bg-secondary);
        z-index: 1;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--vpx-text-secondary);
    }}
    .nbe-{uid} .nbe-error {{
        border: none;
        color: #323130;
        background: #fde7e9;
        padding: 14px 16px;
        font-size: 12px;
        white-space: pre-wrap;
    }}
    .nbe-{uid} .nbe-message {{
        font-size: 12px;
        color: var(--vpx-text-tertiary);
        background: var(--vpx-bg-tertiary);
        padding: 14px 16px;
    }}
    </style>
    """

    html_parts = [f'<div class="nbe-{uid}"><div class="nbe-shell">']
    html_parts.append('<div class="nbe-header">')
    html_parts.append('<h1 class="nbe-title">Notebook Explorer</h1>')
    html_parts.append(f'<div class="nbe-subtitle">{html_module.escape(workspace_name)} - {html_module.escape(workspace_id)}</div>')
    html_parts.append("</div>")

    html_parts.append('<div class="nbe-metrics">')
    for metric in metrics:
        html_parts.append('<div class="nbe-metric">')
        html_parts.append(f'<div class="nbe-metric-value">{html_module.escape(str(metric["value"]))}</div>')
        html_parts.append(f'<div class="nbe-metric-label">{html_module.escape(str(metric["label"]))}</div>')
        html_parts.append("</div>")
    html_parts.append("</div>")

    html_parts.append('<div class="nbe-tabs">')
    for index, tab in enumerate(tabs):
        active = " nbe-active" if index == 0 else ""
        html_parts.append(f'<button class="nbe-tab{active}" data-nbe-tab="{html_module.escape(tab["id"])}">{html_module.escape(tab["label"])}</button>')
    html_parts.append("</div>")

    for index, tab in enumerate(tabs):
        visible = " nbe-visible" if index == 0 else ""
        html_parts.append(f'<div class="nbe-panel{visible}" data-nbe-panel="{html_module.escape(tab["id"])}">')
        html_parts.append('<div class="nbe-grid">')
        for section in tab["sections"]:
            span_class = " nbe-span" if section.get("kind") in {"table", "code", "message"} else ""
            html_parts.append(f'<section class="nbe-section{span_class}">')
            html_parts.append('<div class="nbe-section-head">')
            html_parts.append(f'<h2 class="nbe-section-title">{html_module.escape(section["title"])}</h2>')
            html_parts.append("</div>")
            html_parts.append('<div class="nbe-section-body">')

            kind = section["kind"]
            if kind == "table":
                note = f'{section["row_count"]} row(s)'
                search_id = f"nbe-search-{uid}-{tab['id']}-{section['title']}".replace(" ", "-")
                html_parts.append(
                    '<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--vpx-border);background:var(--vpx-bg-tertiary)">'
                    f'<input class="nbe-search" id="{html_module.escape(search_id)}" placeholder="Filter rows…" />'
                    f'<div class="nbe-note" id="{html_module.escape(search_id)}-count">{html_module.escape(note)}</div>'
                    '</div>'
                )
                html_parts.append('<div class="nbe-table-wrap">')
                html_parts.append(f'<table data-nbe-search-target="{html_module.escape(search_id)}"><thead><tr>')
                for column in section["columns"]:
                    html_parts.append(f"<th>{html_module.escape(str(column)) if str(column) else ''}</th>")
                html_parts.append("</tr></thead><tbody>")
                for row in section["rows"]:
                    html_parts.append("<tr>")
                    for cell in row:
                        html_parts.append(f"<td>{_render_cell_value(cell)}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</tbody></table></div>")
            elif kind == "message":
                html_parts.append(f'<div class="nbe-message">{html_module.escape(section["content"])}</div>')
            else:
                html_parts.append(f'<div class="nbe-error">{html_module.escape(section.get("error", "Unknown error"))}</div>')

            html_parts.append("</div></section>")
        html_parts.append("</div></div>")

    html_parts.append('<div style="padding:10px 24px;font-size:11px;color:var(--vpx-text-tertiary);text-align:right;border-top:1px solid var(--vpx-border);background:var(--vpx-bg-tertiary)">made by jocelyn with &#10084;</div>')
    html_parts.append("</div></div>")

    script = f"""
    <script>
    (function() {{
        var root = document.querySelector('.nbe-{uid}');
        if (!root) return;
        var tabs = Array.from(root.querySelectorAll('.nbe-tab'));
        var panels = Array.from(root.querySelectorAll('.nbe-panel'));

        tabs.forEach(function(tab) {{
            tab.addEventListener('click', function() {{
                var target = tab.getAttribute('data-nbe-tab');
                tabs.forEach(function(item) {{ item.classList.remove('nbe-active'); }});
                panels.forEach(function(panel) {{ panel.classList.remove('nbe-visible'); }});
                tab.classList.add('nbe-active');
                var panel = root.querySelector('[data-nbe-panel="' + target + '"]');
                if (panel) panel.classList.add('nbe-visible');
            }});
        }});

        root.querySelectorAll('.nbe-search').forEach(function(input) {{
            input.addEventListener('input', function() {{
                var table = root.querySelector('table[data-nbe-search-target="' + input.id + '"]');
                if (!table) return;
                var query = input.value.toLowerCase();
                var rows = Array.from(table.querySelectorAll('tbody tr'));
                var shown = 0;
                rows.forEach(function(row) {{
                    var text = row.textContent.toLowerCase();
                    var visible = !query || text.indexOf(query) !== -1;
                    row.style.display = visible ? '' : 'none';
                    if (visible) shown++;
                }});
                var counter = document.getElementById(input.id + '-count');
                if (counter) {{
                    counter.textContent = shown + ' row' + (shown !== 1 ? 's' : '');
                }}
            }});
        }});

        root.querySelectorAll('[data-nbe-iso]').forEach(function(node) {{
            var iso = node.getAttribute('data-nbe-iso');
            if (!iso) return;
            var dt = new Date(iso);
            if (Number.isNaN(dt.getTime())) return;
            node.textContent = dt.toLocaleString();
            node.title = iso;
        }});
    }})();
    </script>
    """

    display(HTML(styles + "".join(html_parts) + script))


@log
def explore_notebooks(workspace: str | UUID, folder: Optional[str | PathLike | UUID] = None, search_string: Optional[str] = None) -> None:
    """
    Opens a notebook UI for inspecting Fabric notebooks.

    Parameters
    ----------
    workspace : str | uuid.UUID
        Fabric workspace name or ID.
    folder : str | os.PathLike | uuid.UUID, default=None
        The folder within the workspace to search.
        Defaults to None which searches the entire workspace.
    search_string : str, default=None
        Optional string to search inside notebooks.
    """

    workspace_name, workspace_id, metrics, tabs = _build_notebook_snapshot(
        workspace=workspace,
        folder=folder,
        search_string=search_string,
    )
    _render_notebook_ui(
        workspace_name=workspace_name,
        workspace_id=workspace_id,
        metrics=metrics,
        tabs=tabs,
    )