import PyPDF2
import pandas as pd
import os
import tabula
import io
import fitz # PyMuPDF
import io
from PIL import Image
import csv


def extract_unformated_text(file_path):
  res = []
  with open(file_path,'rb') as pdf_file:
      read_pdf = PyPDF2.PdfReader(pdf_file)
      number_of_pages = len(read_pdf.pages)
      for page_number in range(number_of_pages):   # use xrange in Py2
          page = read_pdf.pages[page_number] 
          page_content = page.extract_text()
          res.append(page_content)
  rows = zip(res)
  

  with open('C:/Users/Divya/OneDrive/Desktop/teddy/static/doc_images/unformatted_pdf_text_extracted.csv', "w", encoding="utf-8") as f:
      writer = csv.writer(f)
      for row in rows:
          writer.writerow(row)

def extract_unformated_PDF_tables(file_path):
  tables = tabula.read_pdf(file_path, pages="all")
  print("Total tables extracted:", len(tables))
  # iterate over extracted tables and export as excel individually
  folder_name = "C:/Users/Divya/OneDrive/Desktop/teddy/static/doc_images/"
  for i, table in enumerate(tables, start=1):
      table.to_excel(os.path.join(folder_name, f"table_{i}.xlsx"), index=False)


def extract_unformated_PDF_images(file):
# open the file
  pdf_file = fitz.open(file)
  print(file)
  for page_index in range(len(pdf_file)):
      # get the page itself
      page = pdf_file[page_index]
      # get image list
      image_list = page.get_images()
      print("image_list")
      # printing number of images found in this page
      if image_list:
          print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
      else:
          print("[!] No images found on page", page_index)
      for image_index, img in enumerate(image_list, start=1):
          # get the XREF of the image
          xref = img[0]
          # extract the image bytes
          base_image = pdf_file.extract_image(xref)
          image_bytes = base_image["image"]
          # get the image extension
          image_ext = base_image["ext"]
          # load it to PIL
          image = Image.open(io.BytesIO(image_bytes))
          # save it to local disk
          image.save(open(f"C:/Users/Divya/OneDrive/Desktop/teddy/static/doc_images/image{page_index+1}_{image_index}.{image_ext}", "wb"))

def unformatted_pdf_extraction(file):
  extract_unformated_text(file)
  extract_unformated_PDF_tables(file)
  extract_unformated_PDF_images(file)
