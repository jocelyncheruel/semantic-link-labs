import html as html_module
import json
from typing import Any, Optional
from uuid import UUID, uuid4

import pandas as pd
from IPython.display import HTML, display
from sempy._utils._log import log

from sempy_labs.tom import connect_semantic_model


def _is_utc_timestamp(value: str) -> bool:
    """Checks whether a string looks like an ISO 8601 UTC timestamp."""

    return value.endswith("Z") and "T" in value


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
            calc_items = (
                [ci.Name for ci in table.CalculationGroup.CalculationItems]
                if table.CalculationGroup is not None
                else []
            )
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
                    "type": (
                        "Calculation Group"
                        if table.CalculationGroup is not None
                        else (
                            "Calculated Table"
                            if tom.is_calculated_table(table_name=table.Name)
                            else "Table"
                        )
                    ),
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
                var table = root.querySelector(
                    'table[data-smu-search-target="' + input.id + '"]'
                );
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