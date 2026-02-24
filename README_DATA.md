# Data Folder Organization

All CSV files have been organized into the `data` folder for better project structure.

## CSV Files Location

All CSV files should be placed in the `data` folder:
- `data/housing.csv` - Training dataset
- `data/input.csv` - Input file for predictions
- `data/output.csv` - Output file with predictions

## Setup Instructions

1. **Run the organization script** (if CSV files are still in the root directory):
   ```bash
   python organize_data.py
   ```

2. **Or manually move CSV files**:
   - Create a `data` folder if it doesn't exist
   - Move all `.csv` files to the `data` folder

## Updated Code

The following files have been updated to use the `data` folder:
- `main.py` - All CSV file paths now point to `data/` folder
- The code automatically creates the `data` folder if it doesn't exist

## File Structure

```
project-gudgaon/
├── data/
│   ├── housing.csv
│   ├── input.csv
│   └── output.csv
├── main.py
├── app.py
├── organize_data.py
└── ...
```
