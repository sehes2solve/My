/****************************************************************************************/
/*																						*/
/*	Kroenke, Auer, Vandenberg, and Yoder - Database Processing (15th Edition) Chapter 12	*/
/*																						*/
/*	Heather Sweeney Designs Data Warehouse Database Create Tables						*/
/*																						*/
/*	These are the Microsoft SQL Server 2016 code solutions								*/
/*																						*/
/****************************************************************************************/


CREATE TABLE TIMELINE(
		TimeID			Int					NOT NULL,
		Date			Date				NOT NULL,
		MonthID			Int					NOT NULL,
		MonthText		Char(15)			NOT NULL,
		QuarterID		Int					NOT NULL,
		QuarterText		Char(10)			NOT NULL,
		Year			Char(10)			NOT NULL,
		CONSTRAINT		TIMELINE_PK		PRIMARY KEY(TimeID)
		);

CREATE TABLE CUSTOMER(
		CustomerID		Int					NOT NULL,
		CustomerName	Char(75)			NOT NULL,
		EmailDomain		VarChar(100)		NOT NULL,
		PhoneAreaCode	Char(6)				NOT NULL,
		City			Char(35)			NULL,
		State			Char(2)				NULL,
		ZIP				Char(10)			NULL,
		CONSTRAINT 		CUSTOMER_PK 		PRIMARY KEY(CustomerID)
		);

CREATE TABLE PRODUCT(
		ProductNumber	Char(35)			NOT NULL,
		ProductType		Char(25)			NOT NULL,
		ProductName 	VarChar(75)			NOT NULL,
		CONSTRAINT 		PRODUCT_PK			PRIMARY KEY(ProductNumber)
		);

CREATE TABLE PRODUCT_SALES(
		TimeID			Int					NOT NULL,
		CustomerID		Int					NOT NULL,
		ProductNumber	Char(35) 			NOT NULL,
		Quantity		Int					NOT NULL,
		UnitPrice		Numeric(9,2)		NOT NULL,
		Total			Numeric(9,2	)		NULL,
		CONSTRAINT		SALES_PK
		PRIMARY KEY	    (TimeID, CustomerID, ProductNumber),

		CONSTRAINT		PS_TIMELINE_FK FOREIGN KEY(TimeID)
								REFERENCES TIMELINE(TimeID)
										ON UPDATE NO ACTION
										ON DELETE NO ACTION,

		CONSTRAINT		PS_CUSTOMER_FK FOREIGN KEY(CustomerID)
								REFERENCES CUSTOMER(CustomerID)
										ON UPDATE NO ACTION
										ON DELETE NO ACTION,
		CONSTRAINT		PS_PRODUCT_FK FOREIGN KEY(ProductNumber)
								REFERENCES PRODUCT(ProductNumber)
										ON UPDATE NO ACTION
										ON DELETE NO ACTION
		);

 