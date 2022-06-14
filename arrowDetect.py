def arrowDetect(img_arr):
	r = img_arr[0]
	g = img_arr[1]
	b = img_arr[2]
	test_length = 6
	threshold = 2
	test_string = "".join(["1" for i in range(test_length)])
	arrow_detect_strings = []
	for i in range(len(r)):
	    s = ""
	    for j in range(len(r[0])):
	        if b[i][j] > 160 and r[i][j] < 160:
	            s += "1"
	        else:
	            s += "0"
	    print(s)
	    arrow_detect_strings.append(s)

	count = 0
	for string in arrow_detect_strings:
	    if test_string in string:
	        count += 1
	        if count == threshold:
	            return True
	return False