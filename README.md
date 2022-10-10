# Drive-By Coupon EDA

_This is a Solution Practical Application 5.1 exercise of Module 5, in the Berkeley Haas, Professional Certificate in Machine Learning and AI course_

## Will a Customer Accept the Coupon?
### Context

Imagine driving through town and a coupon is delivered to your cell phone for a restaraunt near where you are driving. Would you accept that coupon and take a short detour to the restaraunt? Would you accept the coupon but use it on a sunbsequent trip? Would you ignore the coupon entirely? What if the coupon was for a bar instead of a restaraunt? What about a coffee house? Would you accept a bar coupon with a minor passenger in the car? What about if it was just you and your partner in the car? Would weather impact the rate of acceptance? What about the time of day?

Obviously, proximity to the business is a factor on whether the coupon is delivered to the driver or not, but what are the factors that determine whether a driver accepts the coupon once it is delivered to them? How would you determine whether a driver is likely to accept a coupon?

### Overview

The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon versus those that did not.

### Data

This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. Answers that the user will drive there ‘right away’ or ‘later before the coupon expires’ are labeled as ‘Y = 1’ and answers ‘no, I do not want the coupon’ are labeled as ‘Y = 0’. There are five different types of coupons -- less expensive restaurants (under $20), coffee houses, carry out & take away, bar, and more expensive restaurants ($20 - $50).

### Data Description

The attributes of this data set include:
1. User attributes
    -  Gender: male, female
    -  Age: below 21, 21 to 25, 26 to 30, etc.
    -  Marital Status: single, married partner, unmarried partner, or widowed
    -  Number of children: 0, 1, or more than 1
    -  Education: high school, bachelors degree, associates degree, or graduate degree
    -  Occupation: architecture & engineering, business & financial, etc.
    -  Annual income: less than \\$12500, \\$12500 - \\$24999, \\$25000 - \\$37499, etc.
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater
    than 8
    -  Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or
    greater than 8
    -  Number of times that he/she eats at a restaurant with average expense less than \\$20 per
    person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    

2. Contextual attributes
    - Driving destination: home, work, or no urgent destination
    - Location of user, coupon and destination: we provide a map to show the geographical
    location of the user, destination, and the venue, and we mark the distance between each
    two places with time of driving. The user can see whether the venue is in the same
    direction as the destination.
    - Weather: sunny, rainy, or snowy
    - Temperature: 30F, 55F, or 80F
    - Time: 10AM, 2PM, or 6PM
    - Passenger: alone, partner, kid(s), or friend(s)


3. Coupon attributes
    - time before it expires: 2 hours or one day


### Initial exploration

Data has NaNs in the columns `car`, `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`.

Looking at the column 'car', there are very few respondents and the values that are filled in don't look representative and null values dominate the counts. 

```
----------------------
Column car
----------------------
Null/NaN: 12576
Scooter and motorcycle                      22
Mazda5                                      22
do not drive                                22
crossover                                   21
Car that is too old to install Onstar :D    21
Name: car, dtype: int64
```
So we will drop it and. There are some values in there which are 'Do Not Drive' indicating it belongs to a non-driver. Some questions involve filtering for drivers. Since we are dropping this column, we will assume everyone is a driver.

`Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20` and `Restaurant20To50` have some minimal missing values.
```
----------------------
Column Bar
----------------------
Null/NaN: 107
never    5197
less1    3482
1~3      2473
4~8      1076
gt8       349
Name: Bar, dtype: int64
----------------------
Column CoffeeHouse
----------------------
Null/NaN: 217
less1    3385
1~3      3225
never    2962
4~8      1784
gt8      1111
Name: CoffeeHouse, dtype: int64
----------------------
Column CarryAway
----------------------
Null/NaN: 151
1~3      4672
4~8      4258
less1    1856
gt8      1594
never     153
Name: CarryAway, dtype: int64
----------------------
Column RestaurantLessThan20
----------------------
Null/NaN: 130
1~3      5376
4~8      3580
less1    2093
gt8      1285
never     220
Name: RestaurantLessThan20, dtype: int64
----------------------
Column Restaurant20To50
----------------------
Null/NaN: 189
less1    6077
1~3      3290
never    2136
4~8       728
gt8       264
Name: Restaurant20To50, dtype: int64
```
Looking at the value distributions, we will fill the null values with the median values (highest `value_counts()`), under the assumption that if someone didn't respond, they most probably do what most people do.

Finally in total
```
Accepted        7210
Not Accepted    5474
Name: Coupon Accepted, dtype: int64
```
__56.8% of those given coupons accepted them__

And when looking at the various types of coupons offered:
```
		Coupon Accepted	Percentage	Overall Percentage
coupon	Coupon Accepted			
Carry out & Take away	Accepted	1760	73.547848	13.875749
Restaurant(<20)	Accepted	1970	70.710696	15.531378
Bar	Not Accepted	1190	58.998513	9.381898
Restaurant(20-50)	Not Accepted	834	55.898123	6.575213
Coffee House	Not Accepted	2001	50.075075	15.775781
Accepted	1995	49.924925	15.728477
Restaurant(20-50)	Accepted	658	44.101877	5.187638
Bar	Accepted	827	41.001487	6.520025
Restaurant(<20)	Not Accepted	816	29.289304	6.433302
Carry out & Take away	Not Accepted	633	26.452152	4.990539
```

### EDA of Bar Coupons

#### What proportion of bar coupons were accepted?

```
Not Accepted    1190
Accepted         827
Name: Coupon Accepted, dtype: int64
```
__59% of customers going to bars accepted coupons__

#### Compare the acceptance rate between those who went to a bar 3 or fewer times a month to those who went more

```
	Group	Coupon Accepted	Count	Percentage	Overall Percentage
2	Bar Visits > 3	Accepted	153	76.884422	13.121784
0	Bar Visits <= 3	Accepted	510	52.740434	43.739280
1	Bar Visits <= 3	Not Accepted	457	47.259566	39.193825
3	Bar Visits > 3	Not Accepted	46	23.115578	3.945111
```
- Vast majority of coupons went to those that visit bars 3 or less times a month
- Acceptance rate for those going more than 3 times is significantly higher!

#### Compare the acceptance rate between drivers who go to a bar more than once a month and are over the age of 25 to the all others. Is there a difference?
```
	Group	Coupon Accepted	Count	Percentage	Overall Percentage
0	Bar Visits > 1 & Age <= 25	Accepted	85	64.393939	4.214179
2	Other	Not Accepted	1143	60.636605	56.668319
3	Other	Accepted	742	39.363395	36.787308
1	Bar Visits > 1 & Age <= 25	Not Accepted	47	35.606061	2.330193
```
Those who go to a bar more than once a month and are 25 or less of age are more marginally more inclined to accept coupons

#### Use the same process to compare the acceptance rate between drivers who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry.

```
	Group	Coupon Accepted	Count	Percentage	Overall Percentage
0	Bar Visits >= 1, non-kid Passenger in car and ...	Accepted	110	71.895425	5.453644
2	Other	Not Accepted	1147	61.534335	56.866634
3	Other	Accepted	717	38.465665	35.547843
1	Bar Visits >= 1, non-kid Passenger in car and ...	Not Accepted	43	28.104575	2.131879
```
People going to bars more than once a month, not working in FFF and having a passenger in the car that is not a kid have a significantly higher acceptance rate than others.

#### Compare the acceptance rates between those drivers who:
go to bars more than once a month, had passengers that were not a kid, and were not widowed OR
go to bars more than once a month and are under the age of 30 OR
go to cheap restaurants more than 4 times a month and income is less than 50K

```
Group	Coupon Accepted	Count	Percentage	Overall Percentage
0	Bars > 1, Age < 30	Accepted	227	68.580060	11.254338
2	Bars > 1, No kids, Not Widowed	Accepted	24	63.157895	1.189886
6	RestaurantsLess20 > 4, Income < 50k	Not Accepted	188	54.651163	9.320773
7	RestaurantsLess20 > 4, Income < 50k	Accepted	156	45.348837	7.734259
3	Bars > 1, No kids, Not Widowed	Not Accepted	14	36.842105	0.694100
1	Bars > 1, Age < 30	Not Accepted	104	31.419940	5.156173
```
- Those who go to bars more than once a month and are aged below 30 have the highest acceptance rate
- Those who visit cheap restaurants and have low income have the least acceptance rate

#### Summary
- customers accept coupons for Bars at an average rate of 59%
- Vast majority of coupons went to those that visit Bars 3 or less times a month
- Acceptance rate for those going more than 3 times is significantly higher!
- Acceptance is also higher is the customer is of 25 years of age or less
- And when there are passengers in the car that are not kids
- Those NOT working in Farming, Fishing or Forestry tend to have a slightly higher propensity to accept the coupons
- If the customer tends to spend money on Restaurants, they are less likely to accept Bar coupons, possibly because they don't go to bars as much

### EDA of CarryAway Coupons

From an initial analysis of distribution of coupon acceptance based on number of times a customer does carry away, Acceptance rate is correlated with frequency.

#### What is the overall distribution of coupon acceptance for Carraway customers?
- Those identified as 'female' tend to accept or reject coupons less frequently (indicating it's offered to them less) with a wider variance than those identified as 'male'. Male's tend to accept coupons at a far higher rate
- Those with lower incomes (<62.5K) have a higher propensity to accept coupons with low variance (higher certainty)
- Average acceptance rate of CarryaAway coupons is 73.5%, giving us the benchmark to test various customer attributes against

#### Do kids and Income have any relevance to who gets coupons and accepts them?
```

Coupon Accepted	Percentage
passanger	Coupon Accepted		
Alone	Accepted	1022	42.707898
Friend(s)	Accepted	560	23.401588
Alone	Not Accepted	383	16.005015
Friend(s)	Not Accepted	179	7.480150
Kid(s)	Accepted	107	4.471375
Partner	Accepted	71	2.966987
Kid(s)	Not Accepted	45	1.880485
Partner	Not Accepted	26	1.086502
```
```
Group	Coupon Accepted	Count	Percentage	Overall Percentage
0	Income < 62.5k, Kids in Car	Accepted	66	76.744186	2.758044
2	Income < 62.5k, No Kids in Car	Accepted	1118	75.184936	46.719599
6	Income >=62.5k, No Kids in Car	Accepted	535	70.954907	22.356874
4	Income >=62.5k, Kids in Car	Accepted	41	62.121212	1.713331
5	Income >=62.5k, Kids in Car	Not Accepted	25	37.878788	1.044714
7	Income >=62.5k, No Kids in Car	Not Accepted	219	29.045093	9.151692
3	Income < 62.5k, No Kids in Car	Not Accepted	369	24.815064	15.419975
1	Income < 62.5k, Kids in Car	Not Accepted	20	23.255814	0.835771
```
- Lower incomes get higher coupon offers. Acceptance rate also tends to be marginally higher for lower income levels
- Acceptance rate is significantly boosted if NO passengers are present. If a passenger is present, coupon acceptance is more likely when passenger is not family member
- Low income is a significant factor in determining if coupon is offered. But passengers and passengers being family members should also be considered and given higher weightage

#### How about Gender and Age and destination?
```
	Group	Coupon Accepted	Count	Percentage	Overall Percentage
14	Male, Home, Age >= 36	Accepted	86	90.526316	3.593815
12	Male, Home, Age < 36	Accepted	174	80.930233	7.271208
18	Male, No Urgent Place, Age >= 36	Accepted	153	79.687500	6.393648
4	Female, No Urgent Place, Age < 36	Accepted	204	77.566540	8.524864
2	Female, Home, Age >= 36	Accepted	116	77.333333	4.847472
16	Male, No Urgent Place, Age < 36	Accepted	226	76.094276	9.444212
6	Female, No Urgent Place, Age >= 36	Accepted	163	72.123894	6.811534
0	Female, Home, Age < 36	Accepted	139	72.020725	5.808608
10	Female, Work, Age >= 36	Accepted	108	68.354430	4.513163
20	Male, Work, Age < 36	Accepted	167	67.886179	6.978688
22	Male, Work, Age >= 36	Accepted	69	63.888889	2.883410
8	Female, Work, Age < 36	Accepted	155	62.000000	6.477225
9	Female, Work, Age < 36	Not Accepted	95	38.000000	3.969912
23	Male, Work, Age >= 36	Not Accepted	39	36.111111	1.629753
21	Male, Work, Age < 36	Not Accepted	79	32.113821	3.301295
11	Female, Work, Age >= 36	Not Accepted	50	31.645570	2.089427
1	Female, Home, Age < 36	Not Accepted	54	27.979275	2.256582
7	Female, No Urgent Place, Age >= 36	Not Accepted	63	27.876106	2.632679
17	Male, No Urgent Place, Age < 36	Not Accepted	71	23.905724	2.966987
3	Female, Home, Age >= 36	Not Accepted	34	22.666667	1.420811
5	Female, No Urgent Place, Age < 36	Not Accepted	59	22.433460	2.465524
19	Male, No Urgent Place, Age >= 36	Not Accepted	39	20.312500	1.629753
13	Male, Home, Age < 36	Not Accepted	41	19.069767	1.713331
15	Male, Home, Age >= 36	Not Accepted	9	9.473684	0.376097
```
- Gender plays a huge role. Males tend to pick up food more often
- Folks headed home are more likely to pick up food
- Folks > 35 years of age are more likely to pick up food
__These categories should be prioritized__

#### Summary
- __All customers are assumed to be drivers, even though there were 22 rows where the person indicated they do not drive. This is because the column 'car' had too many NaNs and could not be used__
- Acceptance rate on coupons for CarryAway food is 73.5%
- Frequent CarryAway shoppers will use accept coupons more often
- Males are more likely to use coupons
- Folks traveling home are more likely to accept coupons
- Folks >= 35 years of age are more likely to use coupons
- Folks traveling alone or with friends are more likely to use coupons