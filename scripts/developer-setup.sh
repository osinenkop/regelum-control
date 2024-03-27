SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR
cd ..
echo Entered $(pwd).
echo Installing additional Python dependecies...
python3 -m "pip" install -r requirements-dev.txt
echo Installing other dependecies...
sudo apt install pandoc
echo Installing hooks...
pre-commit install
echo Done!
