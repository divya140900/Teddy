from flask import Flask,request,render_template,redirect, url_for
import os
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from form60 import *
from cheque import *
# from printedcheque import *
from documents import *
app = Flask(__name__)



app.config["IMAGE_UPLOADS"] = "C:/Users/Divya/OneDrive/Desktop/teddy/static/Images"
app.config["PDF_UPLOADS"] = "C:/Users/Divya/OneDrive/Desktop/teddy/static/chequepdf"

#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

def handwrittencheque(chequeimagepath):
	pdf_to_image_for_cheques(chequeimagepath)

# def printedcheque(printedchequeimagepath):
# 	pdf_to_image_for_printed_cheques(printedchequeimagepath)

def documentpdf(docpath):
	unformatted_pdf_extraction(docpath)



@app.route('/home',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']
		option = request.form['options']

		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)


		filename = secure_filename(image.filename)

		if option == "form60":
			jsonname = 'output(copy).json'

			basedir = os.path.abspath(os.path.dirname(__file__))
			image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
			dococr(app.config["IMAGE_UPLOADS"]+"/"+filename)

		if option == 'cheque':
			basedir = os.path.abspath(os.path.dirname(__file__))
			image.save(os.path.join(basedir,app.config["PDF_UPLOADS"],filename))
			handwrittencheque(app.config["PDF_UPLOADS"]+"/"+filename)
			return render_template("main.html",filename=filename)

		# if option == 'printed cheque':
		# 	basedir = os.path.abspath(os.path.dirname(__file__))
		# 	image.save(os.path.join(basedir,app.config["PDF_UPLOADS"],filename))
		# 	printedcheque(app.config["PDF_UPLOADS"]+"/"+filename)
		# 	return render_template("main.html",filename=filename)

		if option == "doc":
			basedir = os.path.abspath(os.path.dirname(__file__))
			image.save(os.path.join(basedir,app.config["PDF_UPLOADS"],filename))
			documentpdf(app.config["PDF_UPLOADS"]+"/"+filename)
			return render_template("main.html",filename=filename)


		# test.clean_text(app.config["IMAGE_UPLOADS"]+"/"+filename)

		return render_template("main.html",filename=filename, option = option)



	return render_template('main.html')



# @app.route('/display/<filename>')
# def display_image(filename):
# 	return redirect(url_for('static',filename = "/Images" + filename), code=301)


app.run(debug=True,port=5000)