#!/bin/zsh
set -e

# Get the directory where the script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set project root, which is two levels up from the script's directory
PROJECT_ROOT=$(dirname $(dirname "${SCRIPT_DIR}"))

# Change to the project root directory to ensure relative paths work as expected
cd "${PROJECT_ROOT}"

echo "--- Starting Data Download and Organisation ---"

# Define paths relative to the project root
DATA_DIR="data"
RAW_DIR="${DATA_DIR}/raw"
URL_FILE="${SCRIPT_DIR}/yolo_data_urls.txt"

# Ensure the raw directory exists
mkdir -p "${RAW_DIR}"

# Check if the raw directory is already populated.
if [ -n "$(ls -A ${RAW_DIR})" ]; then
    echo "Directory '${RAW_DIR}' is not empty. Skipping download and unzip."
    echo "To re-download, please empty the '${RAW_DIR}' directory."
    echo "--- Script Finished ---"
    exit 0
fi

echo "'${RAW_DIR}' is empty. Proceeding with download and extraction."

# Loop through each specified data source in the URL file.
while read -r folder_name url; do
    # Skip empty lines
    if [ -z "$folder_name" ]; then
        continue
    fi

    zip_file="${DATA_DIR}/${folder_name}.zip"
    target_dir="${RAW_DIR}/${folder_name}"

    # If the zip file does not exist, download it.
    if [ ! -f "${zip_file}" ]; then
        echo "Zip file '${zip_file}' not found. Downloading..."
        aria2c -x 16 -s 16 -k 1M --user-agent="Mozilla/5.0" -o "${zip_file}" "${url}"
    else
        echo "Zip file '${zip_file}' found."
    fi

    # If the target directory does not exist, unzip the archive.
    if [ ! -d "${target_dir}" ]; then
        echo "Unzipping '${zip_file}' to '${target_dir}'..."
        # Unzip to a temporary directory first to handle archives that might not have a root folder
        temp_extract_dir=$(mktemp -d)
        unzip -q "${zip_file}" -d "${temp_extract_dir}"

        # If the zip file contains a single directory, move that directory
        # Otherwise, move the contents of the temp directory
        num_top_level_items=$(find "${temp_extract_dir}" -mindepth 1 -maxdepth 1 | wc -l)
        if [ "${num_top_level_items}" -eq 1 ]; then
            extracted_item=$(find "${temp_extract_dir}" -mindepth 1 -maxdepth 1)
            mv "${extracted_item}" "${target_dir}"
            rm -r "${temp_extract_dir}"
        else
            mv "${temp_extract_dir}" "${target_dir}"
        fi
        echo "Successfully created '${target_dir}'."
    else
        echo "Target directory '${target_dir}' already exists. Skipping unzip."
    fi

done < "${URL_FILE}"

echo "--- Data download and organisation complete. ---"
