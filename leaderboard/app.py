import gradio as gr
import pandas as pd
import json
import io

from constants import (
    SUBMIT_INTRODUCTION,
    COLUMN_NAMES,
    MODEL_INFO,
    ALL_RESULTS,
    NEW_DATA_TITLE_TYPE,
    SINGLE_DOMAIN_RESULTS,
    TABLE_INTRODUCTION,
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    COLUMN_NAMES_HUMAN,
    CSV_DIR_HUMAN_DOMAIN_RESULTS,
    CSV_DIR_OPEN_DOMAIN_RESULTS,
    HUMAN_DOMAIN_RESULTS,
    CSV_DIR_SINGLE_DOMAIN_RESULTS,
    TABLE_INTRODUCTION_HUMAN,
    LEADERBORAD_INTRODUCTION,
    OPEN_DOMAIN_RESULTS,
)

global \
    filter_component, \
    data_component_opendomain, \
    data_component_humandomain, \
    data_component_singledomain


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def compute_scores(input_data):
    return [
        None,
        [
            input_data["total_score"],
            input_data["aes_score"],
            input_data["motion_score"],
            input_data["facesim_cur"],
            input_data["gme_score"],
            input_data["nexus_score"],
            input_data["natural_score"],
        ],
    ]


def compute_scores_human_domain(input_data):
    return [
        None,
        [
            input_data["total_score"],
            input_data["aes_score"],
            input_data["motion_score"],
            input_data["facesim_cur"],
            input_data["gme_score"],
            input_data["natural_score"],
        ],
    ]


def add_opendomain_eval(
    input_file,
    model_name_textbox: str,
    revision_name_textbox: str,
    venue_type_dropdown: str,
    team_name_textbox: str,
    model_link: str,
):
    if input_file is None:
        return "Error! Empty file!"
    else:
        selected_model_data = json.load(io.BytesIO(input_file))

        scores = compute_scores(selected_model_data)
        input_data = scores[1]
        input_data = [float(i) for i in input_data]

        csv_data = pd.read_csv(CSV_DIR_OPEN_DOMAIN_RESULTS)

        if revision_name_textbox == "":
            col = csv_data.shape[0]
            model_name = model_name_textbox
            name_list = [
                name.split("]")[0][1:] if name.endswith(")") else name
                for name in csv_data["Model"]
            ]
            assert model_name not in name_list
        else:
            model_name = revision_name_textbox
            model_name_list = csv_data["Model"]
            name_list = [
                name.split("]")[0][1:] if name.endswith(")") else name
                for name in model_name_list
            ]
            if revision_name_textbox not in name_list:
                col = csv_data.shape[0]
            else:
                col = name_list.index(revision_name_textbox)

        if model_link == "":
            model_name = model_name  # no url
        else:
            model_name = "[" + model_name + "](" + model_link + ")"

        venue = venue_type_dropdown
        if team_name_textbox == "":
            team = "User Upload"
        else:
            team = team_name_textbox

        new_data = [
            model_name,
            venue,
            team,
            f"{input_data[0] * 100:.2f}%",
            f"{input_data[1] * 100:.2f}%",
            f"{input_data[2] * 100:.2f}%",
            f"{input_data[3] * 100:.2f}%",
            f"{input_data[4] * 100:.2f}%",
            f"{input_data[5] * 100:.2f}%",
            f"{input_data[6] * 100:.2f}%",
        ]
        csv_data.loc[col] = new_data
        csv_data.to_csv(CSV_DIR_OPEN_DOMAIN_RESULTS, index=False)
    return "Evaluation successfully submitted!"


def add_humandomain_eval(
    input_file,
    model_name_textbox: str,
    revision_name_textbox: str,
    venue_type_dropdown: str,
    team_name_textbox: str,
    model_link: str,
):
    if input_file is None:
        return "Error! Empty file!"
    else:
        selected_model_data = json.load(io.BytesIO(input_file))

        scores = compute_scores_human_domain(selected_model_data)
        input_data = scores[1]
        input_data = [float(i) for i in input_data]

        csv_data = pd.read_csv(CSV_DIR_HUMAN_DOMAIN_RESULTS)

        if revision_name_textbox == "":
            col = csv_data.shape[0]
            model_name = model_name_textbox
            name_list = [
                name.split("]")[0][1:] if name.endswith(")") else name
                for name in csv_data["Model"]
            ]
            assert model_name not in name_list
        else:
            model_name = revision_name_textbox
            model_name_list = csv_data["Model"]
            name_list = [
                name.split("]")[0][1:] if name.endswith(")") else name
                for name in model_name_list
            ]
            if revision_name_textbox not in name_list:
                col = csv_data.shape[0]
            else:
                col = name_list.index(revision_name_textbox)

        if model_link == "":
            model_name = model_name  # no url
        else:
            model_name = "[" + model_name + "](" + model_link + ")"

        venue = venue_type_dropdown
        if team_name_textbox == "":
            team = "User Upload"
        else:
            team = team_name_textbox

        new_data = [
            model_name,
            venue,
            team,
            f"{input_data[0] * 100:.2f}%",
            f"{input_data[1] * 100:.2f}%",
            f"{input_data[2] * 100:.2f}%",
            f"{input_data[3] * 100:.2f}%",
            f"{input_data[4] * 100:.2f}%",
            f"{input_data[5] * 100:.2f}%",
        ]
        csv_data.loc[col] = new_data
        csv_data.to_csv(CSV_DIR_HUMAN_DOMAIN_RESULTS, index=False)
    return "Evaluation successfully submitted!"


def add_singledomain_eval(
    input_file,
    model_name_textbox: str,
    revision_name_textbox: str,
    venue_type_dropdown: str,
    team_name_textbox: str,
    model_link: str,
):
    if input_file is None:
        return "Error! Empty file!"
    else:
        selected_model_data = json.load(io.BytesIO(input_file))

        scores = compute_scores(selected_model_data)
        input_data = scores[1]
        input_data = [float(i) for i in input_data]

        csv_data = pd.read_csv(CSV_DIR_SINGLE_DOMAIN_RESULTS)

        if revision_name_textbox == "":
            col = csv_data.shape[0]
            model_name = model_name_textbox
            name_list = [
                name.split("]")[0][1:] if name.endswith(")") else name
                for name in csv_data["Model"]
            ]
            assert model_name not in name_list
        else:
            model_name = revision_name_textbox
            model_name_list = csv_data["Model"]
            name_list = [
                name.split("]")[0][1:] if name.endswith(")") else name
                for name in model_name_list
            ]
            if revision_name_textbox not in name_list:
                col = csv_data.shape[0]
            else:
                col = name_list.index(revision_name_textbox)

        if model_link == "":
            model_name = model_name  # no url
        else:
            model_name = "[" + model_name + "](" + model_link + ")"

        venue = venue_type_dropdown
        if team_name_textbox == "":
            team = "User Upload"
        else:
            team = team_name_textbox

        new_data = [
            model_name,
            venue,
            team,
            f"{input_data[0] * 100:.2f}%",
            f"{input_data[1] * 100:.2f}%",
            f"{input_data[2] * 100:.2f}%",
            f"{input_data[3] * 100:.2f}%",
            f"{input_data[4] * 100:.2f}%",
            f"{input_data[5] * 100:.2f}%",
            f"{input_data[6] * 100:.2f}%",
        ]
        csv_data.loc[col] = new_data
        csv_data.to_csv(CSV_DIR_SINGLE_DOMAIN_RESULTS, index=False)
    return "Evaluation successfully submitted!"


def get_all_df_opendomain():
    df = pd.read_csv(CSV_DIR_OPEN_DOMAIN_RESULTS)
    df = df.sort_values(by="TotalScore↑", ascending=False)
    return df


def get_baseline_df_opendomain():
    df = pd.read_csv(CSV_DIR_OPEN_DOMAIN_RESULTS)
    df = df.sort_values(by="TotalScore↑", ascending=False)
    present_columns = MODEL_INFO + checkbox_group_opendomain.value
    df = df[present_columns]
    return df


def get_all_df_humandomain():
    df = pd.read_csv(CSV_DIR_HUMAN_DOMAIN_RESULTS)
    df = df.sort_values(by="TotalScore↑", ascending=False)
    return df


def get_baseline_df_humandomain():
    df = pd.read_csv(CSV_DIR_HUMAN_DOMAIN_RESULTS)
    df = df.sort_values(by="TotalScore↑", ascending=False)
    present_columns = MODEL_INFO + checkbox_group_humandomain.value
    df = df[present_columns]
    return df


def get_all_df_singledomain():
    df = pd.read_csv(CSV_DIR_SINGLE_DOMAIN_RESULTS)
    df = df.sort_values(by="TotalScore↑", ascending=False)
    return df


def get_baseline_df_singledomain():
    df = pd.read_csv(CSV_DIR_SINGLE_DOMAIN_RESULTS)
    df = df.sort_values(by="TotalScore↑", ascending=False)
    present_columns = MODEL_INFO + checkbox_group_singledomain.value
    df = df[present_columns]
    return df


block = gr.Blocks()


with block:
    gr.HTML("""
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src="https://www.pnglog.com/6xm07l.png" style='width: 400px; height: auto; margin-right: 10px;' />
        </div>
    """)
    gr.Markdown(LEADERBORAD_INTRODUCTION)
    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        # table Opendomain
        with gr.TabItem("🏅 Open-Domain", elem_id="OpenS2V-Nexus-tab-table", id=0):
            with gr.Row():
                with gr.Accordion("Citation", open=False):
                    citation_button = gr.Textbox(
                        value=CITATION_BUTTON_TEXT,
                        label=CITATION_BUTTON_LABEL,
                        elem_id="citation-button",
                        show_copy_button=True,
                    )

            gr.Markdown(TABLE_INTRODUCTION)

            checkbox_group_opendomain = gr.CheckboxGroup(
                choices=ALL_RESULTS,
                value=OPEN_DOMAIN_RESULTS,
                label="Select options",
                interactive=True,
            )

            data_component_opendomain = gr.components.Dataframe(
                value=get_baseline_df_opendomain,
                headers=COLUMN_NAMES,
                type="pandas",
                datatype=NEW_DATA_TITLE_TYPE,
                interactive=False,
                visible=True,
            )

            def on_checkbox_group_change_opendomain(selected_columns):
                selected_columns = [
                    item for item in ALL_RESULTS if item in selected_columns
                ]
                present_columns = MODEL_INFO + selected_columns
                updated_data = get_baseline_df_opendomain()[present_columns]
                updated_data = updated_data.sort_values(
                    by=present_columns[1], ascending=False
                )
                updated_headers = present_columns
                update_datatype = [
                    NEW_DATA_TITLE_TYPE[COLUMN_NAMES.index(x)] for x in updated_headers
                ]

                filter_component = gr.components.Dataframe(
                    value=updated_data,
                    headers=updated_headers,
                    type="pandas",
                    datatype=update_datatype,
                    interactive=False,
                    visible=True,
                )

                return filter_component

            checkbox_group_opendomain.change(
                fn=on_checkbox_group_change_opendomain,
                inputs=checkbox_group_opendomain,
                outputs=data_component_opendomain,
            )

        # table HumanDomain
        with gr.TabItem("🏅 Human-Domain", elem_id="OpenS2V-Nexus-tab-table", id=1):
            with gr.Row():
                with gr.Accordion("Citation", open=False):
                    citation_button = gr.Textbox(
                        value=CITATION_BUTTON_TEXT,
                        label=CITATION_BUTTON_LABEL,
                        elem_id="citation-button",
                        show_copy_button=True,
                    )

            gr.Markdown(TABLE_INTRODUCTION_HUMAN)

            checkbox_group_humandomain = gr.CheckboxGroup(
                choices=HUMAN_DOMAIN_RESULTS,
                value=HUMAN_DOMAIN_RESULTS,
                label="Select options",
                interactive=True,
            )

            data_component_humandomain = gr.components.Dataframe(
                value=get_baseline_df_humandomain,
                headers=COLUMN_NAMES_HUMAN,
                type="pandas",
                datatype=NEW_DATA_TITLE_TYPE,
                interactive=False,
                visible=True,
            )

            def on_checkbox_group_change_humandomain(selected_columns):
                selected_columns = [
                    item for item in ALL_RESULTS if item in selected_columns
                ]
                present_columns = MODEL_INFO + selected_columns
                updated_data = get_baseline_df_humandomain()[present_columns]
                updated_data = updated_data.sort_values(
                    by=present_columns[1], ascending=False
                )
                updated_headers = present_columns
                update_datatype = [
                    NEW_DATA_TITLE_TYPE[COLUMN_NAMES_HUMAN.index(x)]
                    for x in updated_headers
                ]

                filter_component = gr.components.Dataframe(
                    value=updated_data,
                    headers=updated_headers,
                    type="pandas",
                    datatype=update_datatype,
                    interactive=False,
                    visible=True,
                )

                return filter_component

            checkbox_group_humandomain.change(
                fn=on_checkbox_group_change_humandomain,
                inputs=checkbox_group_humandomain,
                outputs=data_component_humandomain,
            )

        # table SingleDomain
        with gr.TabItem("🏅 Single-Domain", elem_id="OpenS2V-Nexus-tab-table", id=2):
            with gr.Row():
                with gr.Accordion("Citation", open=False):
                    citation_button = gr.Textbox(
                        value=CITATION_BUTTON_TEXT,
                        label=CITATION_BUTTON_LABEL,
                        elem_id="citation-button",
                        show_copy_button=True,
                    )

            gr.Markdown(TABLE_INTRODUCTION)

            checkbox_group_singledomain = gr.CheckboxGroup(
                choices=ALL_RESULTS,
                value=SINGLE_DOMAIN_RESULTS,
                label="Select options",
                interactive=True,
            )

            data_component_singledomain = gr.components.Dataframe(
                value=get_baseline_df_singledomain,
                headers=COLUMN_NAMES,
                type="pandas",
                datatype=NEW_DATA_TITLE_TYPE,
                interactive=False,
                visible=True,
            )

            def on_checkbox_group_change_singledomain(selected_columns):
                selected_columns = [
                    item for item in ALL_RESULTS if item in selected_columns
                ]
                present_columns = MODEL_INFO + selected_columns
                updated_data = get_baseline_df_singledomain()[present_columns]
                updated_data = updated_data.sort_values(
                    by=present_columns[1], ascending=False
                )
                updated_headers = present_columns
                update_datatype = [
                    NEW_DATA_TITLE_TYPE[COLUMN_NAMES.index(x)] for x in updated_headers
                ]

                filter_component = gr.components.Dataframe(
                    value=updated_data,
                    headers=updated_headers,
                    type="pandas",
                    datatype=update_datatype,
                    interactive=False,
                    visible=True,
                )

                return filter_component

            checkbox_group_singledomain.change(
                fn=on_checkbox_group_change_singledomain,
                inputs=checkbox_group_singledomain,
                outputs=data_component_singledomain,
            )

        # table Submission
        with gr.TabItem("🚀 Submit here! ", elem_id="seed-benchmark-tab-table", id=4):
            with gr.Row():
                gr.Markdown(SUBMIT_INTRODUCTION, elem_classes="markdown-text")

            with gr.Row():
                gr.Markdown(
                    "# ✉️✨ Submit your result here!", elem_classes="markdown-text"
                )

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(
                        label="Model name", placeholder="ConsisID"
                    )
                    revision_name_textbox = gr.Textbox(
                        label="Revision Model Name (Optinal)", placeholder="ConsisID"
                    )
                    venue_type_dropdown = gr.Dropdown(
                        label="Venue Type",
                        choices=["Open-Source", "Close-Source"],
                        value="Open-Source",
                    )
                    team_name_textbox = gr.Textbox(
                        label="Your Team Name (If left blank, it will be user upload))",
                        placeholder="User Upload",
                    )
                    model_link = gr.Textbox(
                        label="Model Link",
                        placeholder="https://github.com/PKU-YuanGroup/ConsisID",
                    )

            with gr.Column():
                input_file = gr.File(label="Click to Upload a json File", type="binary")

                submit_button_opendomain = gr.Button("Submit Result (Open-Domain)")
                submit_button_humandomain = gr.Button("Submit Result (Human-Domain)")
                submit_button_singledomain = gr.Button("Submit Result (Single-Domain)")

                submission_result = gr.Markdown()

                submit_button_opendomain.click(
                    add_opendomain_eval,
                    inputs=[
                        input_file,
                        model_name_textbox,
                        revision_name_textbox,
                        venue_type_dropdown,
                        team_name_textbox,
                        model_link,
                    ],
                    outputs=submission_result,
                )

                submit_button_humandomain.click(
                    add_humandomain_eval,
                    inputs=[
                        input_file,
                        model_name_textbox,
                        revision_name_textbox,
                        venue_type_dropdown,
                        team_name_textbox,
                        model_link,
                    ],
                    outputs=submission_result,
                )

                submit_button_singledomain.click(
                    add_singledomain_eval,
                    inputs=[
                        input_file,
                        model_name_textbox,
                        revision_name_textbox,
                        venue_type_dropdown,
                        team_name_textbox,
                        model_link,
                    ],
                    outputs=submission_result,
                )

    with gr.Row():
        data_run = gr.Button("Refresh")
        data_run.click(get_baseline_df_opendomain, outputs=data_component_opendomain)
        data_run.click(get_baseline_df_humandomain, outputs=data_component_humandomain)
        data_run.click(
            get_baseline_df_singledomain, outputs=data_component_singledomain
        )

block.launch()
