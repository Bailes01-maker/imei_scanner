import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
import pillow_heif  # registers HEIC/HEIF in Pillow
import argparse

class BoxScannerSimple:
    # Define the output CSV file path
    OUTPUT_CSV = "results.csv"
    
    def __init__(self):
        # Initialize the scanner by loading existing data or creating a new dataframe
        if os.path.exists(self.OUTPUT_CSV):
            self.df = pd.read_csv(self.OUTPUT_CSV)
        else:
            self.df = pd.DataFrame(columns=["Timestamp", "UPC", "SerialNumber", "IMEI", "EID"])
    
    def load_image(self, path):
        """
        Load an image from the given path, handling both regular images and HEIC/HEIF files.
        Returns the image in a format suitable for OpenCV processing.
        """
        # Handle HEIC/HEIF files explicitly
        if path.lower().endswith(('.heic', '.heif')):
            try:
                # Use pillow_heif to read HEIC files
                heif_file = pillow_heif.read_heif(path)
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                image = image.convert("RGB")
            except Exception as e:
                print(f"Error opening HEIC file: {e}")
                raise
        else:
            # Regular image handling for non-HEIC files
            image = Image.open(path).convert("RGB")
            
        # Convert PIL image to OpenCV format
        arr = np.array(image)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    def preprocess(self, img):
        """
        Preprocess the image to improve OCR accuracy.
        Converts to grayscale, applies blur, and thresholding.
        """
        # Convert to grayscale for OCR processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply binary thresholding using Otsu's method
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    
    def ocr(self, img):
        """
        Perform OCR on the preprocessed image to extract text.
        """
        # Basic configuration for OCR
        config = r'--psm 6'  # Assume a single uniform block of text
        return pytesseract.image_to_string(img, config=config)
    
    def extract(self, text):
        """
        Extract product information (UPC, Serial Number, IMEI, EID) from the OCR text
        using regular expression pattern matching.
        """
        # Print raw text for debugging
        print("Raw OCR text:")
        print(text)
        
        # UPC - Focus on the specific pattern in the OCR text
        upc = []
        # Look for digits in the pattern "5 94 9..." or similar (common in Apple UPCs)
        digits_match = re.search(r'5\s*94\s*9[\d\s]+', text)
        if digits_match:
            # Clean up the matched string to get only digits
            digits = re.sub(r'\D', '', digits_match.group(0))
            if len(digits) >= 12:
                upc = [digits[:12]]  # UPC codes are 12 digits
        
        # Serial Number - Handle various OCR misreads
        sn = []
        # Look for patterns like "Serial No.", "faa No." etc. followed by the serial number
        sn_match = re.search(r'[fSs][\w\s]{0,5}\s*No\.\s*(GNO[0-9A-Z]{7,8})', text)
        if sn_match:
            sn = [sn_match.group(1)]
        # If not found with label, look for the GNO pattern directly (common in Apple serial numbers)
        elif 'GNO' in text:
            gno_match = re.search(r'GNO[0-9A-Z]{7,8}', text)
            if gno_match:
                sn = [gno_match.group(0)]
        
        # IMEI - Look for 15-digit number starting with 35 (common in mobile devices)
        imei = []
        imei_match = re.search(r'\b(35\d{13})\b', text)
        if imei_match:
            imei = [imei_match.group(1)]
        
        # EID - Standard format for Apple EID
        eid = []
        # Look for EID followed by a 32-digit number
        eid_match = re.search(r'EID\s*(8904\d{28})', text)
        if eid_match:
            eid = [eid_match.group(1)]
        # If not found with the EID label, try to find the number directly
        elif '8904' in text:
            eid_direct = re.search(r'(8904\d{28})', text)
            if eid_direct:
                eid = [eid_direct.group(1)]
        
        # Print what was found for debugging
        print(f"Found UPC: {upc}")
        print(f"Found Serial: {sn}")
        print(f"Found IMEI: {imei}")
        print(f"Found EID: {eid}")
        
        return upc, sn, imei, eid
    
    def scan(self, path):
        """
        Main scanning function that processes an image and extracts product information.
        Saves the results to a CSV file and returns the updated dataframe.
        """
        # Load and process the image
        img = self.load_image(path)
        prep = self.preprocess(img)
        text = self.ocr(prep)
        
        # Extract product information
        upc, sn, imei, eid = self.extract(text)
        
        # Create a new row with the extracted information
        row = {
            "Timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "UPC": ";".join(upc),
            "SerialNumber": ";".join(sn),
            "IMEI": ";".join(imei),
            "EID": ";".join(eid),
        }
        
        # Use pandas concat instead of append (which is deprecated)
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.OUTPUT_CSV, index=False)
        
        return self.df


def main():
    """
    Main function that parses command line arguments and runs the scanner.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Scan box images for product information')
    parser.add_argument('image_path', type=str, help='Path to the image file to scan')
    
    args = parser.parse_args()
    
    try:
        # Create scanner and process the image
        scanner = BoxScannerSimple()
        result = scanner.scan(args.image_path)
        
        # Display results
        print("\nScanning complete. Results saved to:", scanner.OUTPUT_CSV)
        print("\nLatest scan results:")
        print(result.tail(1).to_string(index=False))
    except Exception as e:
        # Handle errors and provide troubleshooting tips
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure pillow_heif is properly installed: pip install pillow-heif")
        print("2. Check if the image path is correct")
        print("3. Try converting the HEIC file to JPG using another tool and scan the JPG")


# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
