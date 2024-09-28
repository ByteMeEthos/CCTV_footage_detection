cd data
pip install gdown

mkdir -p SCface
zip_file="scface.zip"

if [ ! -f "$zip_file" ]; then
    echo "$zip_file not found. Downloading..."
    gdown --id 1mxbAgil0-Lbka9FnNTRyrxHKlVB0vgGQ
    echo "Data downloaded"
fi

echo "Unzipping data to SCface folder..."
unzip -o "$zip_file" -d SCface
echo "Data unzipped to SCface folder"