import os                      
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich import box

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
DATA_FOLDER = "data"
DEFAULT_TRAIN_FILE = os.path.join(DATA_FOLDER, "housing.csv")
DEFAULT_INPUT_FILE = os.path.join(DATA_FOLDER, "input.csv")
DEFAULT_OUTPUT_FILE = os.path.join(DATA_FOLDER, "output.csv")

console = Console()


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ]
    )

    return full_pipeline


def train_model(train_path: str = DEFAULT_TRAIN_FILE) -> None:
    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    if not os.path.exists(train_path):
        console.print(
            Panel(
                f"[bold red]❌ Training file not found:[/bold red] '{train_path}'\n\n"
                "[yellow]Please make sure the file exists in the data folder,\n"
                "or run the program again and provide a valid path.[/yellow]",
                title="[bold red]Error[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task1 = progress.add_task("[cyan]📥 Loading training data...", total=None)
        housing = pd.read_csv(train_path)
        progress.update(task1, completed=True)

        task2 = progress.add_task(
            "[cyan]🔧 Preparing data (stratified split by income)...", total=None
        )
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(housing, housing["income_cat"]):
            housing.loc[test_index].drop("income_cat", axis=1).to_csv(
                DEFAULT_INPUT_FILE, index=False
            )
            housing = housing.loc[train_index].drop("income_cat", axis=1)
        progress.update(task2, completed=True)

        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

        num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
        cat_attribs = ["ocean_proximity"]

        task3 = progress.add_task(
            "[cyan]🧱 Building preprocessing pipeline...", total=None
        )
        pipeline = build_pipeline(num_attribs, cat_attribs)
        progress.update(task3, completed=True)

        task4 = progress.add_task(
            "[cyan]⚙️ Transforming data and training model...", total=None
        )
        housing_prepared = pipeline.fit_transform(housing_features)
        model = RandomForestRegressor(random_state=42)
        model.fit(housing_prepared, housing_labels)
        progress.update(task4, completed=True)

        task5 = progress.add_task("[cyan]💾 Saving model and pipeline...", total=None)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)
        progress.update(task5, completed=True)

    # Success message
    success_table = Table(show_header=False, box=None, padding=(0, 2))
    success_table.add_row("✅", "[bold green]Model training complete![/bold green]")
    success_table.add_row("📁", f"[cyan]Model file:[/cyan] [yellow]{MODEL_FILE}[/yellow]")
    success_table.add_row(
        "📁", f"[cyan]Pipeline file:[/cyan] [yellow]{PIPELINE_FILE}[/yellow]"
    )
    success_table.add_row(
        "📄",
        f"[cyan]Example input file created:[/cyan] [yellow]{DEFAULT_INPUT_FILE}[/yellow]",
    )

    console.print(
        Panel(
            success_table,
            title="[bold green]Success[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def run_predictions(
    input_path: str = DEFAULT_INPUT_FILE, output_path: str = DEFAULT_OUTPUT_FILE
) -> None:
    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        console.print(
            Panel(
                "[bold red]❌ No trained model found.[/bold red]\n\n"
                "[yellow]Please train a model first, then run predictions.[/yellow]",
                title="[bold red]Error[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    if not os.path.exists(input_path):
        console.print(
            Panel(
                f"[bold red]❌ Input file not found:[/bold red] '{input_path}'\n\n"
                "[yellow]Make sure the file exists in the data folder or train the model to generate an example.[/yellow]",
                title="[bold red]Error[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task1 = progress.add_task("[cyan]📥 Loading model and pipeline...", total=None)
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        progress.update(task1, completed=True)

        task2 = progress.add_task(
            f"[cyan]📂 Reading input data from '{input_path}'...", total=None
        )
        input_data = pd.read_csv(input_path)
        progress.update(task2, completed=True)

        task3 = progress.add_task(
            "[cyan]⚙️ Applying preprocessing and making predictions...", total=None
        )
        transformed_input = pipeline.transform(input_data)
        predictions = model.predict(transformed_input)
        progress.update(task3, completed=True)

        task4 = progress.add_task(
            "[cyan]💾 Saving predictions to output file...", total=None
        )
        input_data["median_house_value"] = predictions
        input_data.to_csv(output_path, index=False)
        progress.update(task4, completed=True)

    # Success message
    success_table = Table(show_header=False, box=None, padding=(0, 2))
    success_table.add_row(
        "✅", "[bold green]Predictions complete![/bold green]"
    )
    success_table.add_row(
        "📄", f"[cyan]Output file created:[/cyan] [yellow]{output_path}[/yellow]"
    )
    success_table.add_row(
        "📊", f"[cyan]Total predictions made:[/cyan] [yellow]{len(predictions)}[/yellow]"
    )

    console.print(
        Panel(
            success_table,
            title="[bold green]Success[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def prompt_path(prompt_message: str, default: str) -> str:
    user_input = Prompt.ask(
        f"[cyan]{prompt_message}[/cyan]",
        default=default,
        console=console,
    )
    return user_input.strip() or default


def main() -> None:
    # Welcome banner
    welcome_text = Text()
    welcome_text.append("🏡 ", style="bold yellow")
    welcome_text.append("California Housing Price Predictor", style="bold cyan")
    
    console.print()
    console.print(
        Panel(
            welcome_text,
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )
    console.print()
    
    # Description panel
    console.print(
        Panel(
            "[dim]This tool trains a machine learning model and predicts median house values.\n"
            "You don't need any ML background to use it - just follow the simple steps![/dim]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()

    while True:
        # Menu options
        menu_table = Table(show_header=False, box=None, padding=(0, 1))
        menu_table.add_row("[bold green]1[/bold green]", "Train a new model")
        menu_table.add_row("[bold blue]2[/bold blue]", "Run predictions with an existing model")
        menu_table.add_row("[bold red]3[/bold red]", "Exit")

        console.print(
            Panel(
                menu_table,
                title="[bold cyan]What would you like to do?[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        
        choice = Prompt.ask(
            "\n[bold]Enter your choice[/bold]",
            choices=["1", "2", "3"],
            default="1",
            console=console,
        )

        console.print()  # blank line for readability

        if choice == "1":
            console.print(
                Panel(
                    "[yellow]📚 Training Mode[/yellow]\n\n"
                    "[dim]You'll need a CSV file with housing data including 'median_house_value' column.[/dim]",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            console.print()
            train_path = prompt_path(
                "Enter path to training CSV (must include 'median_house_value')",
                DEFAULT_TRAIN_FILE,
            )
            train_model(train_path)
            console.print()  # blank line for readability

        elif choice == "2":
            console.print(
                Panel(
                    "[blue]🔮 Prediction Mode[/blue]\n\n"
                    "[dim]Use your trained model to predict house values for new data.[/dim]",
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )
            console.print()
            input_path = prompt_path(
                "Enter path to input CSV for prediction", DEFAULT_INPUT_FILE
            )
            output_path = prompt_path(
                "Enter desired output CSV filename", DEFAULT_OUTPUT_FILE
            )
            run_predictions(input_path, output_path)
            console.print()

        elif choice == "3":
            console.print(
                Panel(
                    "[bold green]👋 Goodbye![/bold green]\n\n"
                    "[dim]Thanks for using the California Housing Price Predictor![/dim]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
            break

        else:
            console.print(
                Panel(
                    "[bold red]⚠️ Invalid choice[/bold red]\n\n"
                    "[yellow]Please type 1, 2, or 3.[/yellow]",
                    border_style="red",
                    box=box.ROUNDED,
                )
            )
            console.print()


if __name__ == "__main__":
    main()
