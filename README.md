# MedPub Rag Answering Chatbot

## Project Summary
This project is a RAG-based medical question-answering system that uses only PubMed papers to reduce AI hallucinations and provide reliable medical information. Every answer includes the PMID, author, and publication year as evidence. It also uses a CRAG (Corrective RAG) pipeline to evaluate the relevance of retrieved papers and automatically re-search when irrelevant results are found. In addition, the system improves retrieval accuracy by applying multiple query strategies such as Multi-Query, HyDE, and Step-Back.

## Data relationship
![test](assets/DBdiagram.png)

## Methodology 
- **MySQL** : To create a database, load data into tables, and perform various analyses by joining tables.
- **MySQL connect with Python** : To handle and visualize data in order to gain insights.

## Notebook
- Because of the discussion about Folium, you can find the fully preserved notebook here. [San-Francisco-bike-share.ipynb](https://nbviewer.org/github/joanna-jaeeun/San-Francisco-bike-share-analysis/blob/main/San%20Fransico%20Bike%20Share%20Analysis.ipynb)

## Strategies Flow 
All strategies was derived through SQL queries.
- Selecting city : Check the cities in the dataset using Folium, and filter to show only San Francisco.
- Checking usages by weekly and hourly
  
  High usage on weekdays (Monday to Friday), noticeable drop on weekends (Saturday and Sunday)<br>
  Weekday hourly usage trend analyzed in detail.

<p align="center">
  <img src="assets/hourly_usage_trend.png" width="600" alt="Project Logo">
</p>
  
<p align="center">
  <img src="assets/Insight1.jpg" width="600" alt="Project Logo">
</p>

- Fining popular routes (Weekday, commute hours-morining and evening)
  - Defining commute hours to check the distribution
  - Popular routes
    
<div align="center">
  
| start_hour   | end_hour | start_station_name | end_station_name | number |
|--------------|----------|--------------------|------------------|--------|
| 8            | 8        | Harry Bridges Plaza<br>(Ferry Building) | 2nd at Townsend | 1341	|
| 9   | 9     | San Francisco Caltrain 2 <br>(330 Townsend) | Townsend at 7th | 1018 |
| 9   | 9     | Market at Sansome | 2nd at South Park | 991 |

</div>

<div align="center">
  
| start_hour   | end_hour | start_station_name | end_station_name | number |
|--------------|----------|--------------------|------------------|--------|
| 17           | 17        | Embarcadero at Sansome | Steuart at Market | 1064	|
| 17           | 17        | 2nd at Townsend | Harry Bridges Plaza <br>(Ferry Building) | 992 |
| 17           | 17        | 2nd at South Park | Market at Sansome | 911 |

</div>
  
- Proposing redistribution strategies
  
  - Is there enough supply at the start stations of the popular route?
        
    <p align="center">
      <img src="assets/startstation.png" width="600" alt="Project Logo">
    </p>
    <p align="center"><em>Checking Avg_demand, Avg_available, and Demand_Supply_gap at start station</em></p>    
    
  - Identifying nearby bike rental stations within a 1 km radius of popular start stations
  - Business insights
    <p align="center">  
      <img src="assets/Insight2.jpg" width="600" alt="Project Logo">  
    </p>


  - Is there enough capacity at the end stations of the popular route?
    
    <p align="center">
      <img src="assets/endstation.png" width="600" alt="Project Logo">
    </p>
    <p align="center"><em>Checking Avg_arrivals, Remaining_bikes, and Capacity at end station</em></p>    

  - Identifying nearby bike rental stations within a 1 km radius of popular end stations
  
  - Business insights
<p align="center">  
  <img src="assets/Insight3.jpg" width="600" alt="Project Logo">  
</p>


## Conclusions
- User perspective: Reduce user inconvenience caused by bike shortages during commute hours and lack of docking space at destination stations. Provide cost-saving incentives to encourage more frequent and economical use.

- Business perspective: Identify stations with persistent bike shortages or docking surpluses and implement optimized redistribution strategies. Increased user satisfaction is expected to lead to higher overall usage.


## Used Datasets
- SF Bay Area Bike Share [SF Bay Area Bike Share](https://www.kaggle.com/datasets/benhamner/sf-bay-area-bike-share/data)
