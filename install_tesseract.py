import os
import requests
import subprocess
from zipfile import ZipFile
from io import BytesIO

def download_and_install_tesseract():
    # URL for Tesseract installer
    url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    
    print("Downloading Tesseract OCR installer...")
    response = requests.get(url)
    
    if response.status_code == 200:
        installer_path = "tesseract_installer.exe"
        
        # Save the installer
        with open(installer_path, 'wb') as f:
            f.write(response.content)
        
        print("Installing Tesseract OCR...")
        # Run the installer silently
        subprocess.run([installer_path, '/S'])
        
        # Clean up
        os.remove(installer_path)
        print("Tesseract OCR installation completed!")
    else:
        print("Failed to download Tesseract OCR installer")

if __name__ == "__main__":
    download_and_install_tesseract()
