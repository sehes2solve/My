/****************************************************************************************/
/*																						*/
/*	Kroenke, Auer, Vandenberg, and Yoder - Database Processing (15th Edition) Chapter 12	*/
/*																						*/
/*	Heather Sweeney Designs Data Warehouse Database Exercise Solutions							*/
/*																						*/
/*	These are the Microsoft SQL Server 2016 code solutions								*/
/*												Exercise 12.59-L(3, 4)										*/
/****************************************************************************************/

INSERT INTO PAYMENT_TYPE VALUES(1, 'VISA');
INSERT INTO PAYMENT_TYPE VALUES(2, 'MasterCard');
INSERT INTO PAYMENT_TYPE VALUES(3, 'American Express');
INSERT INTO PAYMENT_TYPE VALUES(4, 'Check');
INSERT INTO PAYMENT_TYPE VALUES(5, 'Cash');


/********************************************************************************/
/*																				*/
/*  HSD-DW PRODUCT_SALES Data PaymentType Data									*/
/*																				*/
/*  PRIMARY KEY = (TimeID, CustomerID, ProductNumber)							*/
/*																				*/
/*																				*/
/********************************************************************************/


/*****   Invoice 35000  - '15-Oct-17' = 43023  'RA@somewhere.com' = 3 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43023
		AND		CustomerID = 3
		AND		ProductNumber = 'VK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43023
		AND		CustomerID = 3
		AND		ProductNumber = 'VB001';


/*****   Invoice 35001  - '25-Oct-17' = 43033  'SB@elsewhere.com' = 4 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43033
		AND		CustomerID = 4
		AND		ProductNumber = 'VK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43033
		AND		CustomerID = 4
		AND		ProductNumber = 'VB001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43033
		AND		CustomerID = 4
		AND		ProductNumber = 'BK001';

/*****   Invoice 35002  - '20-Dec-17' = 43089  'SG@somewhere.com' = 7 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43089
		AND		CustomerID = 7
		AND		ProductNumber = 'VK004';

/*****   Invoice 35003  - '25-Mar-18' = 43184  'SB@elsewhere.com' = 4 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43184
		AND		CustomerID = 4
		AND		ProductNumber = 'VK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43184
		AND		CustomerID = 4
		AND		ProductNumber = 'BK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43184
		AND		CustomerID = 4
		AND		ProductNumber = 'VK004';

/*****   Invoice 35004  - '27-Mar-18' = 43186  'KF@somewhere.com' = 6 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186
		AND		CustomerID = 6
		AND		ProductNumber = 'VK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 6
		AND		ProductNumber = 'BK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 6
		AND		ProductNumber = 'VK003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 6
		AND		ProductNumber = 'VB003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 6
		AND		ProductNumber = 'VK004';

/*****   Invoice 35005  - '27-Mar-18' = 43186  'SG@somewhere.com' = 7 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 7
		AND		ProductNumber = 'BK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 7
		AND		ProductNumber = 'BK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 7
		AND		ProductNumber = 'VK003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43186  
		AND		CustomerID = 7
		AND		ProductNumber = 'VK004';

/*****   Invoice 35006  - '31-Mar-18' = 43190  'BP@elsewhere.com' = 9  **********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43190  
		AND		CustomerID = 9
		AND		ProductNumber = 'BK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43190  
		AND		CustomerID = 9
		AND		ProductNumber = 'VK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43190  
		AND		CustomerID = 9
		AND		ProductNumber = 'VB001';

/*****   Invoice 35007  - '03-Apr-18' = 43193  'JT@somewhere.com' = 11 **********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43193  
		AND		CustomerID = 11
		AND		ProductNumber = 'VK003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43193  
		AND		CustomerID = 11
		AND		ProductNumber = 'VB003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43193  
		AND		CustomerID = 11
		AND		ProductNumber = 'VK004';

/*****   Invoice 35008  - '08-Apr-18' = 43198  'SE@elsewhere.com' = 5 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43198  
		AND		CustomerID = 5
		AND		ProductNumber = 'BK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43198  
		AND		CustomerID = 5
		AND		ProductNumber = 'VK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43198  
		AND		CustomerID = 5
		AND		ProductNumber = 'VB001';

/*****   Invoice 35009  - '08-Apr-18' = 43198  'NJ@somewhere.com' = 1 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43198  
		AND		CustomerID = 1
		AND		ProductNumber = 'BK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43198  
		AND		CustomerID = 1
		AND		ProductNumber = 'VK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43198  
		AND		CustomerID = 1
		AND		ProductNumber = 'VB001';

/*****   Invoice 35010  - '23-Apr-18' = 43213  'RA@somewhere.com' = 3 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43213  
		AND		CustomerID = 3
		AND		ProductNumber = 'BK001';

/*****   Invoice 35011  - '07-May-18' = 43227  'BP@elsewhere.com' = 9 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43227
		AND		CustomerID = 9
		AND		ProductNumber = 'VK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43227
		AND		CustomerID = 9
		AND		ProductNumber = 'VB002';

/*****   Invoice 35012  - '21-May-18' = 43241  'SH@elsewhere.com' = 8 ***********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43241  
		AND		CustomerID = 8
		AND		ProductNumber = 'VK003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43241  
		AND		CustomerID = 8
		AND		ProductNumber = 'VB003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43241  
		AND		CustomerID = 8
		AND		ProductNumber = 'VK004';


/********************************************************************************/
/*																				*/
/*   Ralph Able made two purchase on 05-Jun-18.  Since this schema				*/
/*	 in the HSD_DW database	shows product sales summarized for each day		 	*/
/*   - NOT for each sale (Invoice) - the data from these two purchases			*/
/*	 must be combined in a total of all product items Ralph bought on that day   */
/*																				*/
/********************************************************************************/

/*****   Invoice 35013+35016  - '05-Jun-18' = 43256  'RA@somewhere.com' = 3 *****/

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43256  
		AND		CustomerID = 3
		AND		ProductNumber = 'VK001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43256  
		AND		CustomerID = 3
		AND		ProductNumber = 'VB001';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43256  
		AND		CustomerID = 3
		AND		ProductNumber = 'VK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43256  
		AND		CustomerID = 3
		AND		ProductNumber = 'VB002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 1
	WHERE		TimeID = 43256  
		AND		CustomerID = 3
		AND		ProductNumber = 'BK002';

/*****   Invoice 35014  - '05-Jun-18' = 43256  'JT@somewhere.com' = 11 **********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 11
		AND		ProductNumber = 'VK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 11
		AND		ProductNumber = 'VB002';

/*****   Invoice 35015  - '05-Jun-18' = 43256  'JW@elsewhere.com' = 12 **********/

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 12
		AND		ProductNumber = 'VK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 12
		AND		ProductNumber = 'VB003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 12
		AND		ProductNumber = 'BK002';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 12
		AND		ProductNumber = 'VK003';

UPDATE PRODUCT_SALES SET PaymentTypeID = 2
	WHERE		TimeID = 43256  
		AND		CustomerID = 12
		AND		ProductNumber = 'VK004';


/********************************************************************************/
