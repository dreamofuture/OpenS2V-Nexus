MODEL_INFO = ["Model", "Venue", "Evaluated by"]

ALL_RESULTS = [
    "TotalScore↑",
    "Aesthetics↑",
    "Motion↑",
    "FaceSim↑",
    "GmeScore↑",
    "NexusScore↑",
    "NaturalScore↑",
]

OPEN_DOMAIN_RESULTS = [
    "TotalScore↑",
    "Aesthetics↑",
    "Motion↑",
    "FaceSim↑",
    "GmeScore↑",
    "NexusScore↑",
    "NaturalScore↑",
]
HUMAN_DOMAIN_RESULTS = [
    "TotalScore↑",
    "Aesthetics↑",
    "Motion↑",
    "FaceSim↑",
    "GmeScore↑",
    "NaturalScore↑",
]
SINGLE_DOMAIN_RESULTS = [
    "TotalScore↑",
    "Aesthetics↑",
    "Motion↑",
    "FaceSim↑",
    "GmeScore↑",
    "NexusScore↑",
    "NaturalScore↑",
]

NEW_DATA_TITLE_TYPE = [
    "markdown",
    "markdown",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
]

CSV_DIR_OPEN_DOMAIN_RESULTS = "./file/results_Open-Domain.csv"
CSV_DIR_HUMAN_DOMAIN_RESULTS = "./file/results_Human-Domain.csv"
CSV_DIR_SINGLE_DOMAIN_RESULTS = "./file/results_Single-Domain.csv"

COLUMN_NAMES = MODEL_INFO + ALL_RESULTS
COLUMN_NAMES_HUMAN = MODEL_INFO + HUMAN_DOMAIN_RESULTS

LEADERBORAD_INTRODUCTION = """
    # OpenS2V-Eval Leaderboard
    
    Welcome to the leaderboard of the OpenS2V-Eval!
     
    🏆 OpenS2V-Eval is a core component of **OpenS2V-Nexus**, designed to establish a foundational infrastructure for *Subject-to-Video* (S2V) generation. It presents 180 prompts spanning seven major categories of S2V, incorporating both real and synthetic test data. To better align evaluation with human preferences, it introduce three new automatic metrics—NexusScore, NaturalScore, and GmeScore—that independently assess subject consistency, naturalness, and textual relevance in generated videos.
    
    If you like our project, please give us a star ⭐ on GitHub for the latest update.

    [GitHub](https://github.com/PKU-YuanGroup/OpenS2V-Nexus) | [Arxiv](https://arxiv.org/) | [Home Page](https://pku-yuangroup.github.io/OpenS2V-Nexus/) | [OpenS2V-Eval](https://huggingface.co/datasets/BestWishYsh/OpenS2V-Eval) | [OpenS2V-5M](https://huggingface.co/datasets/BestWishYsh/OpenS2V-5M)
"""

SUBMIT_INTRODUCTION = """# Submission Guidelines
    1. Fill in *'Model Name'* if it is your first time to submit your result **or** Fill in *'Revision Model Name'* if you want to update your result.
    2. Fill in your home page to *'Model Link'*.
    3. After evaluation, follow the guidance in the [github repository](https://github.com/PKU-YuanGroup/OpenS2V-Nexus) to obtain `model_name.json` and upload it here.
    4. Click the *'Submit Eval'* button.
    5. Click *'Refresh'* to obtain the updated leaderboard.
"""

TABLE_INTRODUCTION = """In the table below, we summarize each task performance of all the models.
        We use Aesthetic, Motion, FaceSim, GmeScore, NexusScore, and NaturalScore as the primary evaluation metric for each tasks.
    """

TABLE_INTRODUCTION_HUMAN = """In the table below, we summarize each task performance of all the models.
        We use Aesthetic, Motion, FaceSim, GmeScore, and NaturalScore as the primary evaluation metric for each tasks.
    """

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@article{
}"""
