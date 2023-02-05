from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
import numpy as np
import json
import shutil
import os

def align_images(image, template, maxFeatures=500, keepPercent=0.2,
	debug=False):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	# use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)
    	# sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	# check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		cv2.imshow("Matched Keypoints", matchedVis)
		cv2.waitKey(0)
    	# allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
    	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	# return the aligned image
	return aligned

# 	images[i].save('template'+ str(i) +'.jpg', 'JPEG')

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def dococr(imagepath):
    imagepath2 = 'C:/Users/Divya/OneDrive/Desktop/teddy/static/Images/finall_test_page-0001.jpg'
    OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
        "filter_keywords"])
    # define the locations of each area of the document we wish to OCR

    OCR_LOCATIONS = [
        OCRLocation("1", (237, 372, 853, 38),
            ["First", "Name"]),
        OCRLocation("1.1", (237, 409, 853, 38),
            ["Middle", "Nam"]),
        OCRLocation("1.2", (237, 447, 853, 38),
            ["Surname"]),

        OCRLocation("2", (237, 485, 853, 38),
            ["Date", "of", "Birth", "Incorporation", "of", "declarant", "D", "M", "Y"]),

        OCRLocation("3", (237, 561, 853, 38),
            ["First", "Name"]),
        OCRLocation("3.1", (237, 598, 853, 38),
            ["Middle", "Name"]),
        OCRLocation("3.2", (237, 637, 853, 38),
            ["Surname"]),

        OCRLocation("4", (237, 676, 340, 76),
            ["Flat", "Room", "No."]),

        OCRLocation("5", (616, 675, 474, 76),
            ["Floor", "No."]),

        OCRLocation("6", (237, 751, 340, 76),
            ["Name", "of", "premises"]),

        OCRLocation("7", (615, 752, 475, 76),
            ["Block", "Name", "No."]),

        OCRLocation("8", (237, 827, 340, 76),
            ["Road", "Street", "Lane"]),

        OCRLocation("9", (615, 827, 475, 77),
            ["Area", "Locality"]),

        OCRLocation("10", (237, 903, 340, 77),
            ["Town", "City"]),

        OCRLocation("11", (615, 904, 219, 75),
            ["District"]),

        OCRLocation("12", (871, 905, 219, 74),
            ["State"]),

        OCRLocation("13", (236, 980, 133, 75),
            ["Pin", "code"]),

        OCRLocation("14", (405, 979, 393, 76),
            ["Telephone", "Number"]),

        OCRLocation("15", (834, 980, 256, 74),
            ["Mobile", "Number"]),

        OCRLocation("16", (237, 1056, 853, 37),
            ["Amount", "of", "transaction"]),

        OCRLocation("17", (237, 1093, 852, 37),
            ["Date", "of", "transaction"])

    ]
    # load the input image and template from disk
    print("[INFO] loading images...")
    image = cv2.imread(imagepath)
    template = cv2.imread(imagepath2)
    height, width, channels = image.shape
    print(height, width, channels)
    
    # align the images
    print("[INFO] aligning images...")
    aligned = align_images(image, template)

    # initialize a results list to store the document OCR parsing results
    print("[INFO] OCR'ing document...")
    parsingResults = []
    # loop over the locations of the document we are going to OCR
    for loc in OCR_LOCATIONS:
        # extract the OCR ROI from the aligned image
        (x, y, w, h) = loc.bbox
        roi = aligned[y:y + h, x:x + w]
        # OCR the ROI using Tesseract
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb)
    
        # break the text into lines and loop over them
        for line in text.split("\n"):
            # if the line is empty, ignore it
            if len(line) == 0:
                continue
            # convert the line to lowercase and then check to see if the
            # line contains any of the filter keywords (these keywords
            # are part of the *form itself* and should be ignored)
            lower = line.lower()
            count = sum([lower.count(x) for x in loc.filter_keywords])
            # if the count is zero then we know we are *not* examining a
            # text field that is part of the document itself (ex., info,
            # on the field, an example, help text, etc.)
            if count == 0:
                # update our parsing results dictionary with the OCR'd
                # text if the line is *not* empty
                parsingResults.append((loc, line))

    # initialize a dictionary to store our final OCR results
    results = {}
    myresults = {}
    # loop over the results of parsing the document
    for (loc, line) in parsingResults:
        # grab any existing OCR result for the current ID of the document
        r = results.get(loc.id, None)
        # if the result is None, initialize it using the text and location
        # namedtuple (converting it to a dictionary as namedtuples are not
        # hashable)
        if r is None:
            results[loc.id] = (line, loc._asdict())
        # otherwise, there exists an OCR result for the current area of the
        # document, so we should append our existing line
        else:
            # unpack the existing OCR result and append the line to the
            # existing text
            (existingText, loc) = r
            text = "{} | {}".format(existingText, line)
            # update our results dictionary
            results[loc["id"]] = (text, loc)
    # loop over the results
    for (locID, result) in results.items():
        # unpack the result tuple
        (text, loc) = result
        # display the OCR result to our terminal
        # print(loc["id"])
        # print("=" * len(loc["id"]))
        # print("{}\n\n".format(text))
        myresults[loc["id"]] = "{}".format(text)

    json_object = json.dumps(myresults, indent=4)
    print(json_object)
    with open("output.json", "w") as outfile:
        outfile.write(json_object)
    source = 'C:/Users/Divya/OneDrive/Desktop/teddy/output.json'
    destination = 'C:/Users/Divya/OneDrive/Desktop/teddy/static/Json/output(copy).json'
    shutil.copy2(source, destination)
    
