**ESTIMATING GROUND-LEVEL PM2.5 IN NIGERIA FROM SATELLITE AEROSOL DATA AND METEOROLOGY USING MACHINE LEARNING**

**NAME:** OLUWATUYI, OLUWATIMILEHIN EZEKIEL

**INTERN ID:** DF-2026-091

**ABSTRACT**

Fine particulate matter (PM2.5) is among the leading environmental risk factors for premature death. Inhalation triggers oxidative stress, widespread inflammation, and vascular dysfunction. In Nigeria, the limited availability and uneven distribution of ground-based air quality monitoring stations make it difficult to assess the spatial and temporal variations in PM₂.₅ concentrations. Even though the satellite measurements give good geographical coverage, the aerosol optical depth (AOD) is not directly related to ground-level PM₂.₅ concentration due to the effects of atmospheric and meteorological conditions. The aim of this study is to build machine learning models that can estimate daily ground-level PM₂.₅ concentration in selected major cities in Nigeria from satellite AOD and meteorological parameters. In this study, a quantitative methodology will be adopted, considering Lagos, Abuja, and Port Harcourt cities, depending on the availability of appropriate measurement data. The ground-level PM₂.₅ data will be provided by OpenAQ, while the satellite AOD data will be taken from the MODIS Multi-Angle Implementation of Atmospheric Correction (MAIAC) dataset. The duration of the first phase of the study will run from January 2021 to December 2025\. The datasets will be combined and used to analyse the variation in PM₂.₅ concentrations spatially and seasonally. Two algorithms namely, Random Forest and Extreme Gradient Boosting (XGBoost) will be designed to determine daily PM₂.₅ concentrations, and the accuracy of the algorithms will be determined based on ground measurement and statistics, such as R², RMSE, MAE, and MBE. The proposed research will provide useful information regarding the spatiotemporal pattern of PM₂.₅ and the applicability of machine learning in the determination of particulate pollution in Nigeria.

**1.0	INTRODUCTION**

Urbanization and modernization have contributed to the increase in air pollution, which presents serious dangers to people's health and environmental conditions. The topic of air pollution has been investigated in numerous studies because it is an environmental problem of great importance to today's world as well as a great health problem. Air pollutants can be classified into two categories: gaseous air pollutants and particulate matter (PM) which are responsible for the deterioration of air quality. Gaseous air pollutants include carbon monoxide (CO), sulfur dioxide (SO₂), Nitrogen Oxides (NOₓ), Ozone (O₃), and volatile organic compounds (VOCs). Particulate matters, on the other hand, consist of solid particles and liquid droplets suspended in the atmosphere (Singh *et al.*, 2025). Particulate matter can be divided into three basic types according to their aerodynamic diameter. They include PM₁₀ (the aerodynamic diameter is less than 10 µm), PM₂.₅ (aerodynamic diameter is less than 2.5 µm) and ultrafine particulate matter (particles with an aerodynamic diameter of less than 0.1 µm). The effects of these particulate matter are defined by several factors including size, chemical composition, origin, solubility, and ability to produce reactive oxygen species (ROS) (Corneillie *et al.*, 2026).

Ambient fine particulate matter (PM₂.₅) PM₂.₅ remains a major atmospheric pollutant of concern due to its small size, prolonged suspension in the atmosphere, and ability to penetrate deeply into the respiratory system (Xing *et al.*, 2016). It is regarded as one of the biggest environmental risk factors for early deaths in the world. Fine particles have the capability to reach deep into the lung airways; however, some particles or components of particles can reach the blood stream, thus resulting in heart disease, lung diseases, poor birth outcomes and even early mortality (Landrigan *et al.*, 2018). The burden is concentrated in low- and middle-income countries, and sub-Saharan Africa has seen exposure rise alongside urbanisation, traffic growth, diesel generator use, open waste burning and biomass burning (Fisher *et al.*, 2021). According to the guidelines from WHO (2021), 24-hour and annual mean PM₂.₅ concentrations should be 15 and 5 µg/m³, respectively.

The PM₂.₅ levels in the ambient air in Nigeria result from the effect of anthropogenic and natural atmospheric phenomenon. The sources include vehicular exhaust fumes, industries, diesel fuel generators, biomass combustion, waste burning and road dust re-suspension in the atmosphere (Feyisetan *et al.*, 2025). Moreover, the seasonal transfer of Saharan dust during the harmattan season causes an increase in the number of aerosols in the atmosphere especially in the northern part of Nigeria. The sources of the aerosols may differ between urban areas, peri-urban and rural areas, leading to geographical and seasonal variations of particulate matter pollution. Meteorological parameters such as temperature, humidity, wind speed, rainfall and atmospheric mixing affect the distribution of the aerosols. Although reference-grade monitoring instruments provide reliable local measurements, their installation, operation, and maintenance are expensive. There is an opportunity for leveraging recent low-cost sensor technologies to enhance monitoring; however, the measurements from such sensors can be affected by factors like relative humidity, aging of sensors, calibration of sensors, and aerosol characteristics. According to Barkjohn *et al.* (2021), proper correction techniques can enhance the accuracy of PM₂.₅ measurements collected from low-cost optical sensors. However, there are few and scattered monitoring stations on the ground in Nigeria.

Satellite remote sensing is an important complementary technology to monitor atmospheric aerosols over large geographical domains. Aerosol optical depth (AOD) represents the attenuation of solar radiation by aerosols throughout the atmospheric column and is widely used as an indicator of atmospheric aerosol loading. The Moderate Resolution Imaging Spectroradiometer (MODIS), through the Multi-Angle Implementation of Atmospheric Correction (MAIAC) algorithm obtain AOD data at 1 km resolution (Lyapustin *et al.*, 2018). However, the satellite-based AOD does not reflect surface PM₂.₅ levels as the link between columnar aerosol loading and surface particulate levels depends on the atmospheric boundary layer height, relative humidity, aerosol composition, and vertical particle distribution. For example, AOD may be high due to a dust or smoke layer being brought in higher layers in the atmosphere without increasing the levels of surface particulates (Han *et al.*, 2015). Such complexities prevent AOD usage for ground level PM₂.₅ estimation and demonstrate a need for modelling techniques incorporating additional atmospheric and environmental variables.

The application of machine learning algorithms offers possibilities to overcome these obstacles through modeling of nonlinear interaction among satellite-derived AODs, meteorological factors, and geographical parameters. The Random Forest (RF) algorithm, proposed by Breiman (2001), utilizes several decision trees to increase the robustness of predictions and accuracy. In turn, the Extreme Gradient Boosting (XGBoost) algorithm, proposed by Chen and Guestrin (2016), models complex interaction among predictor variables by using a regularised gradient boosting framework. Both of these algorithms can be applied for the purpose of estimating ground-level concentrations of PM₂.₅ from various environmental data sets. Estimation of PM₂.₅ in Nigeria has been investigated in previous studies that explored satellite- and reanalysis-based estimation of PM₂.₅ as well as used atmospheric data and machine learning algorithms (Sani *et al.*, 2026). However, it is still difficult to estimate ground-level concentrations of PM₂.₅ across Nigeria due to several reasons. However, variations in predictive performance depending on geographic locations and environmental conditions pose crucial issues concerning accuracy and generalization. In this respect, this study aim to combine satellite-based AOD and meteorological data with machine learning algorithms to predict PM₂.₅ concentrations in Nigeria focusing on urban cities including Abuja, Lagos and Port Harcourt, compare the predictive performance of machine learning algorithms, and examine their ability to offer accurate predictions under different geographic conditions.

**2.0	LITERATURE REVIEW**

Satellite remote sensing techniques have become a more popular approach for supplementing observations on the ground, especially where air quality monitoring systems are poorly developed. Aerosol optical depth (AOD), which stands for aerosol extinction over the entire atmospheric column, is often applied for calculating surface PM₂.₅ concentrations. Wang and Christopher (2003) studied the association between the MODIS aerosol optical thickness and ground-level PM₂.₅ concentrations measured at seven stations in Alabama, USA. Their study reported a correlation coefficient of 0.70 between satellite aerosol observations and surface PM₂.₅, which increased to above 0.90 when monthly averages were considered. Although these findings demonstrated the potential of satellite observations for particulate pollution assessment, the authors acknowledged that atmospheric conditions influence the relationship between column aerosol loading and surface concentrations. This limitation was further addressed by van Donkelaar *et al.* (2016), who integrated satellite AOD, atmospheric chemical transport simulations, and ground observations using geographically weighted regression to estimate global PM₂.₅ concentrations. Their model achieved a cross-validated coefficient of determination (R²) of 0.81. However, greater uncertainties were observed in mineral dust-affected regions with limited ground monitoring coverage, demonstrating the importance of accounting for geographical and atmospheric differences in satellite-based estimation.

Development in satellite products and meteorological reanalysis data has further opened up avenues for enhancing PM₂.₅ estimates. The MODIS Multi-Angle Implementation of Atmospheric Correction (MAIAC) algorithm was discussed by Lyapustin *et al.* (2018), which generates aerosol data at 1 km spatial resolution. This data can be combined with meteorological data generated through reanalysis data like ERA5, which has been reported by Hersbach *et al.* (2020). Meteorological data provides information regarding atmospheric processes affecting surface particulate concentrations such as humidity-dependent aerosol growth, wind-induced transport, and atmospheric vertical mixing. However, the relationship between satellite AOD and ground-level PM₂.₅ is complicated since high aerosol concentration may exist in the atmosphere without an increase in ground-level concentrations. Therefore, the additional predictive power of meteorological information needs to be empirically evaluated.

In African, Adong *et al.* (2025) compared Multivariate Linear Regression, Random Forest, and multilayer perceptron models for predicting PM₂.₅ at ground level in Kampala, Uganda based on the MODIS MAIAC AOD data and monitoring instrumentations both low-cost and high-resolution grade. Random Forest model showed the best result with RMSE of 7.72 µg/m³, MAE of 5.5 µg/m³, and R² of 0.72. Such results indicate the promising application of machine learning for PM₂.₅ prediction in Africa. However, the study was based on the observations of only two monitoring stations; therefore, the generalization power of its results is limited. Recently, Sani *et al.* (2026) assessed the estimation of surface PM₂.₅ in Nigeria through Random Forest algorithm that combines the ground-based observations, satellite data, atmospheric composition variables, meteorological data, and MERRA-2 reanalysis data. Leave-location-out spatial cross-validation was used and improved results have been obtained in comparison with MERRA-2, although predictive power was different across the monitoring locations.

The reliability of these types of models also relies heavily on the accuracy of ground-based measurements as well as the techniques used in measuring their predictive power. Barkjohn *et al.* (2021) assessed PM₂.₅ measurements using PurpleAir low-cost sensors across the U.S. and found that corrections helped in aligning the measurements with the reference-grade measurements, especially after taking into account the environmental variables such as the relative humidity. This finding will be useful in studies where low-cost sensors have been used to generate training and validation measurements, since uncorrected measurement errors can affect the model predictions. Apart from the measurement errors, Roberts *et al.* (2017) showed that the traditional random cross-validation technique could result in underestimating the errors in prediction when the environmental observations are spatially or temporally correlated. In addition, Meyer and Pebesma (2021) defined the concept of the area of applicability, emphasizing that prediction becomes less reliable in areas where the environmental characteristics differ greatly from the training data.

Generally, the reviewed literature indicates that there exist viable methods for estimating ground-level PM₂.₅ concentration based on satellite-derived AOD, meteorological variables, and machine learning. Nonetheless, while previous research has confirmed the usefulness of Random Forest algorithm and use of multiple data sources, the questions regarding the relative efficiency of other methods, advantages of adding meteorological variables to AOD-based models, and season and monitoring environment-dependent changes in model's effectiveness persist. It should be noted that, for example, Sani *et al.* (2026) has already provided a spatially verified machine learning prediction in Nigeria. Therefore, the proposed study does not purport to present this technique in Nigeria for the first time, but rather investigate the spatiotemporal features of observed PM₂.₅ concentration in selected monitoring locations of Nigeria, its relation to satellite AOD and meteorological variables, and performance of Random Forest and XGBoost compared to AOD and meteorology corrected linear regression model in terms of predicting daily ground-level PM₂.₅ concentrations. The proposed study will additionally examine the effectiveness of the models on monitoring locations, which were not included in training.

**2.1	Statement of the Research Problem**

Ground-level PM₂.₅ estimation is a key input to air quality evaluation and decision-making regarding environmental issues in Nigeria. Nonetheless, ground observations are limited and confined to selected sites; furthermore, cheap sensors could be subject to environmental biases. Machine learning represents one solution in this situation, but the predictive validity might become questionable with non-representative training set or overlapping training/test samples. The main issue, then, is to find an approach to estimate daily ground-level PM₂.₅ based on satellite aerosols and meteorology in Nigeria.

**2.2	Research Questions**

This study seeks to answer the following research questions:

i. 	What are the spatiotemporal characteristics of available ground-level PM₂.₅ concentrations across selected major cities in Nigeria?

ii. 	How can satellite-derived aerosol optical depth (AOD) data and meteorological variables be integrated into a machine learning model to estimate ground-level PM₂.₅ concentrations in Nigeria?

iii. 	How accurately does the developed machine learning model estimate ground-level PM₂.₅ concentrations when evaluated against available ground-based measurements?

**2.3	Research Objectives**

The main objective of this study is to develop and evaluate machine learning models for estimating daily ground-level PM₂.₅ in Nigeria from satellite-derived AOD and meteorological variables.

The specific objectives are to:

i.	examine the spatiotemporal characteristics of available ground-level PM2.5 concentrations across selected major cities in Nigeria;

ii.	develop a machine learning model for estimating ground-level PM2.5 concentrations in Nigeria by integrating satellite-derived aerosol optical depth (AOD) data and meteorological variables; and

iii.	evaluate the performance and predictive accuracy of the developed machine learning model in estimating ground-level PM2.5 concentration using available ground-based measurements.

**3.0	METHODOLOGY**

**3.1	Research Design**

The study will adopt a quantitative, remote sensing-based research design integrating ground-based air quality measurements, satellite-derived aerosol optical depth (AOD), meteorological reanalysis data, and machine learning techniques to estimate daily ground-level fine particulate matter (PM₂.₅) concentrations in Lagos, Abuja, and Port Harcourt, Nigeria.

**3.2	Study Area**

Nigeria is located in West Africa, roughly between latitudes 4° and 14° N, and longitudes 3° to 15° E. Nigeria has diverse climatic zones, namely, tropical rainforest, savanna, and arid. Nigeria features considerable variability in terms of precipitation, humidity, atmospheric dust content, urbanization, and emission sources. Lagos and Abuja along with other selected locations from the list of confirmed ground-based monitoring stations will be used for the study. The coverage is the goal rather than an assumption.

**3.3	Data Sources**

Ground-level PM2.5 concentration data will be collected from the OpenAQ platform for monitoring stations in Lagos, Abuja, and Port Harcourt. Data will be collected through the OpenAQ API, where the observation dates, measurement frequency, completeness, and availability will first be considered for each city to determine the study period. The primary filtering period will be from 1 January 2021 to 31 December 2025, following which the common period with adequate observations for all three cities will be determined. Daily PM2.5 concentration will be preferred since the objective of the study is to model daily concentrations.

AOD will be collected from the MODIS satellite through the NASA MCD19A2 Version 6.1 MAIAC algorithm, which provides daily observations of AOD at an approximate spatial resolution of 1 km. Variables related to meteorology, including air temperature, wind speed, precipitation, and boundary layer height, will be gathered from ECMWF's ERA5 reanalysis dataset through the Copernicus Climate Data Store. Satellite and reanalysis data will be collected for the confirmed OpenAQ study period and at a temporal resolution corresponding to ground observations.

**3.4	Summary of Datasets**

| Dataset | Meaning | Source | Temporal Coverage and Resolution |
| ----- | ----- | ----- | ----- |
| Ground-level PM₂.₅ | Observed concentration of fine particulate matter (µg/m³); target variable | OpenAQ (Lagos, Abuja, and Port Harcourt) | Initial screening: 1 January 2021–31 December 2025; final period determined by verified common coverage |
| Aerosol Optical Depth (AOD) | Measure of aerosol extinction throughout the atmospheric column | NASA MODIS MCD19A2 Version 6.1 MAIAC | Same dates as OpenAQ; daily observations |
| Air temperature | Atmospheric thermal conditions influencing aerosol formation and dispersion | ECMWF ERA5 reanalysis | Same study period; hourly data aggregated to daily means |
| Wind speed | Atmospheric movement influencing pollutant transport and dispersion | ECMWF ERA5, derived from 10 m wind components | Same study period; daily mean |
| Precipitation | Rainfall influencing particle removal through wet deposition | ECMWF ERA5 reanalysis | Same study period; daily total |
| Boundary-layer height | Vertical extent of the atmospheric mixing layer affecting pollutant dilution | ECMWF ERA5 reanalysis | Same study period; daily mean |

**3.5	Data Preprocessing and Spatiotemporal Analysis**

The datasets will be harmonised by observation date and geographical location. Where hourly PM₂.₅ measurements are available, daily means will be calculated subject to adequate data completeness. Daily meteorological variables will be derived from the corresponding ERA5 observations, while MODIS AOD pixels corresponding to the monitoring locations will be extracted using GIS or Google Earth Engine. Missing observations, outliers, and AOD quality flags will be examined before model development. Descriptive statistics, temporal trend analysis, and GIS-based visualisation will be used to assess variations in PM₂.₅ concentrations across the selected cities.

**3.6	Machine Learning Model Development**

Ground-level PM₂.₅ will serve as the dependent variable, while MODIS-derived AOD, air temperature, wind speed, precipitation, and boundary-layer height will serve as predictor variables. Random Forest and XGBoost algorithms will be developed and compared to identify the most suitable model for estimating daily PM₂.₅ concentrations. The dataset will be divided into training and testing subsets using a time-aware validation strategy to minimise data leakage and assess the model's ability to predict observations beyond the training period.

**3.7	Model Evaluation and Spatial Estimation**

Model performance will be evaluated using the coefficient of determination (\\(R^2\\)), root mean square error (RMSE), mean absolute error (MAE), and mean bias error (MBE). Predicted concentrations will be compared with independent ground-based PM₂.₅ measurements to assess predictive accuracy and bias. The best-performing model will subsequently be applied to available daily AOD and meteorological data to generate spatial estimates and maps of ground-level PM₂.₅ concentrations across the selected cities.

**4.0	EXPECTED CONTRIBUTIONS**

The study is expected to make the following contributions:

1\.	This study is expected to provide evidence on the spatial and seasonal variations in ground-level PM₂.₅ concentrations across selected major cities in Nigeria using available monitoring observations.

2\.	demonstrate a framework for combining satellite-derived AOD, meteorological variables, and ground-based PM₂.₅ measurements for air pollution estimation; and

3\.	develop machine learning models for estimating daily ground-level PM₂.₅ concentrations using satellite and meteorological information.

**5.0	TIMELINE**

| S/N | Research Activities |  Timeline |
| ----- | ----- | ----- |
| 1 | Finalisation of research proposal, methodology, and literature review | Oct. 19–25, 2026 |
| 2 | Identification of more eligible monitoring stations and acquisition of ground-level PM₂.₅ measurements from OpenAQ | Oct. 26–Nov. 1, 2026 |
| 3 | Acquisition of satellite-derived MODIS MAIAC AOD and ERA5 meteorological datasets | Nov. 2–8, 2026 |
| 4 | Data cleaning | Nov. 9–22, 2026 |
| 5 | Exploratory analysis of spatial and seasonal PM₂.₅ patterns and relationships with environmental predictors | Nov. 23–29, 2026 |
| 6 | Development of AOD-only and meteorology baseline models | Nov. 30–Dec. 6, 2026 |
| 7 | Development of Random Forest and XGBoost models | Dec. 7–20, 2026 |
| 8 | Model evaluation using spatially independent validation and performance metrics | Dec. 21–27, 2026 |
| 9 | Assessment of seasonal prediction errors and model performance across monitoring environments | Dec. 28, 2026–Jan. 3, 2027 |
| 10 | Interpretation of results and preparation of charts, tables, and figures | Jan. 4–10, 2027 |
| 11 | Writing of results, discussion, conclusions, and recommendations | Jan. 11–24, 2027 |
| 12 | Revision, proofreading, and final submission | Jan. 25–31, 2027 |

**REFERENCES**

Adong, P., Bainomugisha, E., & Dev, S. (2025). Evaluating machine learning methods for PM₂.₅ estimation using satellite AOD, low-cost and reference-grade monitors in Kampala. *International Journal of Environmental Science and Technology*, 22, 15747–15756. https\://doi.org/10.1007/s13762-025-06674-0

Barkjohn, K. K., Gantt, B., & Clements, A. L. (2021). Development and application of a United States-wide correction for PM₂.₅ data collected with the PurpleAir sensor. *Atmospheric Measurement Techniques*, 14, 4617–4637. https\://doi.org/10.5194/amt-14-4617-2021

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. https\://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). https\://doi.org/10.1145/2939672.2939785

Corneillie, L., De Jonge, K., & Laverge, J. (2026). Refining particulate matter classification: Addressing health risks of smaller fractions. Indoor Environments, Article 100164\.

Feyisetan, O. A., Mustapha, S., Ndamitso, M. M., Salau, R. B., Salihu, S. O., Abulude, F. O., & Mathew, J. T. (2025). Spatio-seasonal variability and source inference of PM₂.₅ in Abuja, Nigeria: Implications for urban air quality policy. *Journal of Chemistry and Allied Sciences*, 1(2), 23–35.

Han, Y., Wu, Y., Wang, T., Zhuang, B., Li, S., & Zhao, K. (2015). Impacts of elevated-aerosol-layer and aerosol type on the correlation of AOD and particulate matter with ground-based and satellite measurements in Nanjing, southeast China. *Science of the Total Environment*, 532, 195–207.

Hersbach, H., Bell, B., Berrisford, P., *et al.* (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https\://doi.org/10.1002/qj.3803

Landrigan, P. J., Fuller, R., Acosta, N. J. R., *et al.* (2018). The Lancet Commission on pollution and health. *The Lancet*, 391(10119), 462–512. https\://doi.org/10.1016/S0140-6736(17)32345-0

Lyapustin, A., Wang, Y., Korkin, S., & Huang, D. (2018). MODIS Collection 6 MAIAC algorithm. *Atmospheric Measurement Techniques*, 11, 5741–5765. https\://doi.org/10.5194/amt-11-5741-2018

Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633. https\://doi.org/10.1111/2041-210X.13650

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https\://doi.org/10.1111/ecog.02881

Sani, M., Houdou, A., & Sa'id, R. S. (2026). Machine learning-based estimation of surface PM₂.₅ in Nigeria using integration of satellite, ground observations and reanalysis data. *FUDMA Journal of Sciences*, 10(13), 177–191. https\://doi.org/10.33003/fjs-2026-1013-5627

Singh, V., Srivastava, R. K., & Bhatt, A. K. (2025). Major air pollutants. In V. Singh, R. K. Srivastava, & A. K. Bhatt (Eds.), Battling air and water pollution (pp. 31–42). *Springer Nature Singapore*. https\://doi.org/10.1007/978-981-96-4375-2\_3

van Donkelaar, A., Martin, R. V., Brauer, M., Hsu, N. C., Kahn, R. A., Levy, R. C., Lyapustin, A., Sayer, A. M., & Winker, D. M. (2016). Global estimates of fine particulate matter using a combined geophysical-statistical method with information from satellites, models, and monitors. *Environmental Science & Technology*, 50(7), 3762–3772. https\://doi.org/10.1021/acs.est.5b05833

Wang, J., & Christopher, S. A. (2003). Intercomparison between satellite-derived aerosol optical thickness and PM₂.₅ mass: Implications for air quality studies. *Geophysical Research Letters*, 30(21), 2095\. https\://doi.org/10.1029/2003GL018174

World Health Organization. (2021). WHO global air quality guidelines: Particulate matter (PM₂.₅ and PM₁₀), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide. https\://www\.who.int/publications/i/item/9789240034228

Xing, Y.-F., Xu, Y.-H., Shi, M.-H., & Lian, Y.-X. (2016). The impact of PM2.5 on the human respiratory system. Journal of Thoracic Disease, 8(1), E69–E74. https\://doi.org/10.3978/j.issn.2072-1439.2016.01.19

**Publication Link**

[**https\://medium.com/@oluwatuyitimilehin/weeks-4-and-5-two-rejections-one-verified-topic-and-a-proposal-i-almost-gave-up-on-868eeb87b0a2?sharedUserId=oluwatuyitimilehin**](https://medium.com/@oluwatuyitimilehin/weeks-4-and-5-two-rejections-one-verified-topic-and-a-proposal-i-almost-gave-up-on-868eeb87b0a2?sharedUserId=oluwatuyitimilehin)

