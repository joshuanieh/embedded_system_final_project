def arrowDetect(img_arr):
	test_length = 6
	threshold = 2
	test_string = "".join(["1" for i in range(test_length)])
	arrow_detect_strings = []
	for i in range(len(img_arr)):
	    s = ""
	    for j in range(len(img_arr[0])):
	        if img_arr[i][j][2] > 160 and img_arr[i][j][0] < 160:
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