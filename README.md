## **Capstone Module 3 - Hotel Booking Cancellation Prediction**
By Rully Hilman Simeon

Table of Contents:
   - Business Problem
   - Data Understanding
   - Data preprocessing
   - Modeling
   - Conclusion
   - Recommendation


### **Business Problem**

**Context**
Hotel's revenue depends on the booking of each guest and groups. Booking could come as a direct booking to the hotel, a walk in guest, conventional travel agent or an online travel agent. Each sources and guest could be given a different prices depending on the demand on each dates. Time of the year, events, supply of rooms in each city all affects the demand of the hotel booking.
By predicting the cancellation possibility of a a guest we could forecast the rate of cancellation and maximize ADR by having an overbook

**Problem Statement**
One of the challenges experienced by hoteliers is booking cancellation. Knowing an extimate of booking cancellations could prevent hotels loss of revenue and increase revenue at the same time. During a low season, a booking cancellation means that hotel needs to refund a certain percantage of the room rate. While, on a high season, knowing an estimate of booking cancellation could give hotels an overbooking opportunity. Overbooking is a situation where hotel sold more rooms that it has on its inventory. 

**Goals**
The goal of this machine learning project is to find the most suitable model in predicting the cancellation probability of a booking. Models can be used in hotels in revenue management department as well as for the front office team, or even to enhance the demand forecast research itself.

Cancellation Policy
Guest could have cancelable bookings or a non-refundable bookings. A non-refundable booking could not be cancelled and if the guest cancelled, there would not be any monetary compensation. Cancellable Bookings could be cancelled with certain rules. Different Hotels could have different rules.Big Chain Hotel usually have a 24 - 48 hour cancellation policy. Busier hotel may apply a 72 hour cancellation policy.
Different time frame might resulted in different cancellation fee charged to customers. Knowing the cancelation rate or prediction of guests lets hotel know on how much lost revenue they would experience and it will give hotels time and opportunity to take pre-emptive measures or even for an additional revenue. 

**Analytic Apprroach**
First of all, features/columns that are available will be analyzed. Patterns that affects the degree of possiblity in booking cancellation are the aim. Using a classification algorithm like Random Forest, XGBOOST and KNNeighbors.

**Metric Evaluation**
Evaluation Metrics that will be used are Accuracy, Precision, Recall, F1 Score.

Cancel = Positive  
Type I Error (False Positive) - Accept a false null hypothesis (Guest did not cancel but is predicted otherwise)

Type II Error (False Negative) - Rejecting a true null hypothesis (Guest cancelled but is predicted otherwise)

The aim is to know the cancellation rate (True Positive) and avoiding Type I and II error as much as we can as they would cause losses. 
Therefore a high Recall value, without sacrificing a low precision value would be the aim. Balancing both recall and precision would be favorable. 

### **2. Data Understanding**

The data is obtained from kaggle (https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) which contains detailed booking information in a city hotel or resort hotel such as length of stay, dates, number of guest, booking type, etc.

There are 119390 rows × 32 columns in the data set that will be used in the machine learning model. In this project, classification machine learning will be perform. Predicting a booking whether it will be cancelled or no. 

Columns/Features and its meaning

- Hotel - type of hotel (city or resort)
- is_canceled (Target) - whether the booking is canceled or not
- lead_time
- arrival_date_year
- arrival_date_month 
- arrival_date_week_number
- arrival_date_day_of_month
- stays_in_weekend_nights
- stays_in_week_nights
- adults - number of adult
- children - number of kids
- babies - Number of babies
- meal - Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner)
- country - country of origin / nationality
- market_segment - Tour Agent or Tour Operator
- distribution_chanel
- is repeated guest - whether guest has came before or not
- previous cancellation - whether guest has previously cancelled a booking or not
- previous bookings not cancelled - previous successful booking
- reserved room type - room that the guest booked
- assigned room type - assigned room to the booking
- booking changes - ammendment made to the booking
- deposit_type - In case no payments were found the value is “No Deposit; If the payment was equal or exceeded the total cost of stay, the value is set as “Non Refund”; Otherwise the value is set as “Refundable”
- agent - Travel Agent ID special number
- company - company/corporate that made the booking
- days_in_waiting_list - number of days the customer book to booking confirmed
- customer_type - Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking
- adr - stands for average daily rate (sum of all transaction / total staying nights)
- required_car_parking_spaces - number of parking space the customer requires
- total_of_special_requests - number of special request made by customer
- reservation_status
- reservation_status_date

![Cancellation](https://user-images.githubusercontent.com/99156512/169983821-0fba5e81-7d54-4a8b-baeb-1da912351df2.png)

![year of booking](https://user-images.githubusercontent.com/99156512/169983926-406f3083-fd6f-4f0b-aca8-6ee51188eab3.png)

![Hotel Types](https://user-images.githubusercontent.com/99156512/169983965-0e30f5b6-622c-44d2-bb8d-fadc2d05044c.png)

![month](https://user-images.githubusercontent.com/99156512/169984027-958151e8-8068-4725-a20a-544312f5d15d.png)

![Country of Origin](https://user-images.githubusercontent.com/99156512/169984069-06a7cf72-0a52-45a7-a4b7-216a163c3073.png)

![Market Segment Types](https://user-images.githubusercontent.com/99156512/169984099-bbc89ec8-a76a-4e0f-9335-f89392fbd798.png)


### **3. Data Preprocessing**
![Data Details](https://user-images.githubusercontent.com/99156512/169984148-d7c26572-ae18-4780-8782-3c4919562a9a.png)
![Data Details2](https://user-images.githubusercontent.com/99156512/169984161-e661536c-0ab2-4d7d-bab5-8c26bbbbff21.png)

![Heatmap Missing Value](https://user-images.githubusercontent.com/99156512/169984176-90724c8a-c7f0-4947-ae51-2fc5c71932f2.png)
Heatmap Visualization of the Missing Value 

![crammers](https://user-images.githubusercontent.com/99156512/169984204-a3f2e9d3-4bab-4e0b-a406-09e6e26f856a.png)
![correlation](https://user-images.githubusercontent.com/99156512/169984192-ba563d23-0d9b-4b8c-a09b-f5348d9fdfaf.png)
Correlation with Crammer's V for categorical features and Spearman for numerical features

![dropped](https://user-images.githubusercontent.com/99156512/169985550-8669079f-211a-4a35-85fe-97630d5330cb.png)
Dropping the Unsuitable Features

### **4. Modelling**
![pipeline](https://user-images.githubusercontent.com/99156512/169984235-02b8d06d-e272-4b72-8f8c-8134525d7fc5.png)

![first try](https://user-images.githubusercontent.com/99156512/170025546-62b8d9a8-e7d6-4d03-b358-88abfe2b862f.png)

### **5. Hyper Parameter Tuning**

![RF Tuned](https://user-images.githubusercontent.com/99156512/170025502-c5227948-ae39-4890-a536-1c72277fe904.png)
![ww](https://user-images.githubusercontent.com/99156512/170038181-04f7adfc-f261-4186-8efa-e64e67e268bc.png)

![xgb tuned](https://user-images.githubusercontent.com/99156512/170025479-2c98352a-84aa-4374-a8e8-2f040131a5be.png)
![qq](https://user-images.githubusercontent.com/99156512/170038174-595c1664-fbbe-47fd-9097-314566c20dfc.png)


### **6. Conclusion**
The first three fitting of three models (XGBoost, Random Forest, and Decision Tree) resulted in a relatively good matrics. 
The highest was done with Random Forest with and the second was done with XGBoost.

              Accuracy	  Recall	Precision	
Train RF	|0.996325	  |0.993951	    | 0.996119

Test RF	    |0.889229	  |0.799774	    | 0.890035

Train DT	|  0.996325	  | 0.992905	|  0.997161

Test DT	    | 0.842114	  | 0.792086    |  0.783932

Train XGB	| 0.887396	  | 0.810566	|  0.876169

Test XGB	| 0.871430	  |0.783267	    |  0.857320

So these two models were picked to be further tuned to obtain a better result. 
However a hyperparameter tuning does not guarantee a better result. It resulted in lower value than the initial fittings. 

Thus far, with all the fittings of each algorithms and tunings that had been done. The first Random Forest algorithm model has the highest value of Recall and precision.

### **7. Recommendation**
There are several ways to improve the model on predicting hotel booking cancellations. The most important thing in building this model is the data itself. 
Therefore, to acheive a better model certain steps needs to be taken:

- Although there are 30 columns/features in the dataset, not all the features are relevant or correlated to the target. A more relevant and correlated features to the target needs to be collected.
- The unrelated features should also be dropped. Features that are causing error rather than giving patterns to the target
- The tuned models resulted in a lower evaluation matrixes and to improve this situation a different algorithms needs to be tested 
- Other than new algorithm, improving current model needs another combination of hyper parameter tuning. Taking into account the parameter used suitable to the current computation power. 
