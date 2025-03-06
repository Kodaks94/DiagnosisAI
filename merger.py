from PyPDF2 import PdfMerger

# List of PDF files to merge
folder = "C:/Users/Mahra/Desktop/"
pdfs = [folder+"converted_image.pdf", folder+"Dear Kiana.pdf"]

# Create a PdfMerger object
merger = PdfMerger()

# Append files
for pdf in pdfs:
    merger.append(pdf)

# Save the merged PDF
merger.write("Happy Valentines Day.pdf")
merger.close()

print("PDFs merged successfully!")