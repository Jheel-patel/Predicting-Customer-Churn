# Predicting-Customer-Churn

## Background
Customer attrition is a big issue in any industry. Not surprisingly, one of the major focus of a data scientist is to reduce customer attrition and increase customer retention. It is relatively easier to predict and detect in the industries where monthly billing service exists Eg: telecom, internet, streaming service etc. From an organizational perspective, it is always cheaper to retain existing customers than to acquire new customers.

## Goal
My goal is to build a model to predict whether a customer will churn or not based on given a dataset in this project.

## Data
This data set consists of 100 variables and approx 100 thousand records. It contains different variables explaining the telecom industry's attributes and various factors considered important while dealing with customers of the telecom industry. The target variable here is churn, which explains whether the customer will churn or not. We can use this data set to predict the customers who would churn or who wouldn't churn, depending on various variables available.


<img src="https://s16353.pcdn.co/wp-content/uploads/2018/06/Churn.png" width="500">


|  Continuous Variables  |  Explanation  |
| :---: | :---: |
|  ADJMOU  |  Billing adjusted total minutes of use over the life of the customer  |
|  ADJQTY  |  Billing adjusted total number of calls over the life of the customer  |
|  ADJREV  |  Billing adjusted total revenue over the life of the customer  |
|  ATTEMPT_MEAN  |  Mean number of attempted calls  |
|  ATTEMPT_RANGE  |  Range of number of attempted calls  |
|  AVG3MOU  |  Average monthly minutes of use over the previous three months  |
|  AVG3QTY  |  Average monthly number of calls over the previous three months  |
|  AVG3REV  |  Average monthly revenue over the previous three months  |
|  AVG6MOU  |  Average monthly minutes of use over the previous six months  |
|  AVG6QTY  |  Average monthly number of calls over the previous six months  |
|  AVG6REV  |  Average monthly revenue over the previous six months  |
|  AVGMOU  |  Average monthly minutes of use over the life of the customer  |
|  AVGQTY  |  Average monthly number of calls over the life of the customer  |
|  AVGREV  |  Average monthly revenue over the life of the customer  |
|  BLCK_DAT_MEAN  |  Mean number of blocked (failed) data calls  |
|  BLCK_DAT_RANGE  |  Range of number of blocked (failed) data calls  |
|  BLCK_VCE_MEAN  |  Mean number of blocked (failed) voice calls  |
|  BLCK_VCE_RANGE  |  Range of number of blocked (failed) voice calls  |
|  CALLFWDV_MEAN  |  Mean number of call forwarding calls  |
|  CALLFWDV_RANGE  |  Range of number of call forwarding calls  |
|  CALLWAIT_MEAN  |  Mean number of call waiting calls  |
|  CALLWAIT_RANGE  |  Range of number of call waiting calls  |
|  CC_MOU_MEAN  |  Mean unrounded minutes of use of customer care (see CUSTCARE_MEAN) calls  |
|  CC_MOU_RANGE  |  Range of unrounded minutes of use of customer care calls  |
|  CCRNDMOU_MEAN  |  Mean rounded minutes of use of customer care calls  |
|  CCRNDMOU_RANGE  |  Range of rounded minutes of use of customer care calls  |
|  CHANGE_MOU  |  Percentage change in monthly minutes of use vs previous three month average  |
|  CHANGE_REV  |  Percentage change in monthly revenue vs previous three month average  |
|  COMP_DAT_MEAN  |  Mean number of completed data calls  |
|  COMP_DAT_RANGE  |  Range of number of completed data calls  |
|  COMP_VCE_MEAN  |  Mean number of completed voice calls  |
|  COMP_VCE_RANGE  |  Range of number of completed voice calls  |
|  COMPLETE_MEAN  |  Mean number of completed calls  |
|  COMPLETE_RANGE  |  Range of number of completed calls  |
|  CUSTCARE_MEAN  |  Mean number of customer care calls  |
|  CUSTCARE_RANGE  |  Range of number of customer care calls  |
|  DA_MEAN  |  Mean number of directory assisted calls  |
|  DA_RANGE  |  Range of number of directory assisted calls  |
|  DATOVR_MEAN  |  Mean revenue of data overage  |
|  DATOVR_RANGE  |  Range of revenue of data overage  |
|  DROP_BLK_MEAN  |  Mean number of dropped or blocked calls  |
|  DROP_BLK_RANGE  |  Range of number of dropped or blocked calls  |
|  DROP_DAT_MEAN  |  Mean number of dropped (failed) data calls  |
|  DROP_DAT_RANGE  |  Range of number of dropped (failed) data calls  |
|  DROP_VCE_MEAN  |  Mean number of dropped (failed) voice calls  |
|  DROP_VCE_RANGE  |  Range of number of dropped (failed) voice calls  |
|  EQPDAYS  |  Number of days (age) of current equipment  |
|  INONEMIN_MEAN  |  Mean number of inbound calls less than one minute  |
|  INONEMIN_RANGE  |  Range of number of inbound calls less than one minute  |
|  IWYUS_VCE_MEAN  |  Mean number of inbound wireless to wireless voice calls  |
|  IWYLIS_VCE_RANGE  |  Range of number of inbound wireless to wireless voice calls  |
|  MONTHS  |  Total number of months in service  |
|  MOU_CDAT_MEAN  |  Mean unrounded minutes of use of completed data calls  |
|  MOU_CDAT_RANGE  |  Range of unrounded minutes of use of completed data calls  |
|  MOU_CVCE_MEAN  |  Mean unrounded minutes of use of completed voice calls  |
|  MOU_CVCE_RANGE  |  Range of unrounded minutes of use of completed voice calls  |
|  MOU_MEAN  |  Mean number of monthly minutes of use  |
|  MOU_OPKD_MEAN  |  Mean unrounded minutes of use of off-peak data calls  |
|  MOU_OPKD_RANGE  |  Range of unrounded minutes of use of off-peak data calls  |
|  MOU_OPKV_MEAN  |  Mean unrounded minutes of use of off-peak voice calls  |
|  MOU_OPKV_RANGE  |  Range of unrounded minutes of use of off-peak voice calls  |
|  MOU_PEAD_MEAN  |  Mean unrounded minutes of use of peak data calls  |
|  MOU_PEAD_RANGE  |  Range of unrounded minutes of use of peak data calls  |
|  MOU_PEAV_MEAN  |  Mean unrounded minutes of use of peak voice calls  |
|  MOU_PEAV_RANGE  |  Range of unrounded minutes of use of peak voice calls  |
|  MOU_RANGE  |  Range of number of minutes of use  |
|  MOU_RVCE_MEAN  |  Mean unrounded minutes of use of received voice calls  |
|  MOU_RVCE_RANGE  |  Range of unrounded minutes of use of received voice calls  |
|  MOUIWYLISV_MEAN  |  Mean unrounded minutes of use of inbound wireless to wireless voice calls  |
|  MOUIWYLISV_RANGE  |  Range of unrounded minutes of use of inbound wireless to wireless voice calls  |
|  MOUOWYLISV_MEAN  |  Mean unrounded minutes of use of outbound wireless to wireless voice calls  |
|  MOUOWYLISV_RANGE  |  Range of unrounded minutes of use of outbound wireless to wireless voice calls  |
|  OWYLIS_VCE_MEAN  |  Mean number of outbound wireless to wireless voice calls  |
|  OWYLIS_VCE_RANGE  |  Range of number of outbound wireless to wireless voice calls  |
|  OPK_DAT_MEAN  |  Mean number of off-peak data calls  |
|  OPK_DAT_RANGE  |  Range of number of off-peak data calls  |
|  OPK_VCE_MEAN  |  Mean number of off-peak voice calls  |
|  OPK_VCE_RANGE  |  Range of number of off-peak voice calls  |
|  OVRMOU_MEAN  |  Mean overage minutes of use  |
|  OVRMOU_RANGE  |  Range of overage minutes of use  |
|  OVRREV_MEAN  |  Mean overage revenue  |
|  OVRREV_RANGE  |  Range of overage revenue  |
|  PEAK_DAT_MEAN  |  Mean number of peak data calls  |
|  PEAK_DAT_RANGE  |  Range of number of peak data calls  |
|  PEAK_VCE_MEAN  |  Mean number of inbound and outbound peak voice calls  |
|  PEAK_VCE_RANGE  |  Range of number of inbound and outbound peak voice calls  |
|  PLCD_DAT_MEAN  |  Mean number of attempted data calls placed  |
|  PLCD_DAT_RANGE  |  Range of number of attempted data calls placed  |
|  PLCD_VCE_MEAN  |  Mean number of attempted voice calls placed  |
|  PLCD_VCE_RANGE  |  Range of number of attempted voice calls placed  |
|  RECY_SMS_MEAN  |  Mean number of received SMS calls  |
|  RECV_SMS_RANGE  |  Range of number of received SMS calls  |
|  RECV_VCE_MEAN  |  Mean number of received voice calls  |
|  RECV_VCE_RANGE  |  Range of number of received voice calls  |
|  RETDAYS  |  Number of days since last retention call  |
|  REV_MEAN  |  Mean monthly revenue (charge amount)  |
|  REV_RANGE  |  Range of revenue (charge amount)  |
|  RMCALLS  |  Total number of roaming calls  |
|  RMMOU  |  Total minutes of use of roaming calls  |
|  RMREV  |  Total revenue of roaming calls  |
|  ROAM_MEAN  |  Mean number of roaming calls  |
|  ROAM_RANGE  |  Range of number of roaming calls  |
|  THREEWAY_MEAN  |  Mean number of three way calls  |
|  THREEWAY_RANGE  |  Range of number of three way calls  |
|  TOTCALLS  |  Total number of calls over the life of the customer  |
|  TOTMOU  |  Total minutes of use over the life of the customer  |
|  TOTMRC_MEAN  |  Mean total monthly recurring charge  |
|  TOTMRC_RANGE  |  Range of total monthly recurring charge  |
|  TOTREV  |  Total revenue  |
|  UNAN_DAT_MEAN  |  Mean number of unanswered data calls  |
|  UNAN_DAT_RANGE  |  Range of number of unanswered data calls  |
|  UNAN_VCE_MEAN  |  Mean number of unanswered voice calls  |
|  UNAN_VCE_RANGE  |  Range of number of unanswered voice calls  |
|  VCEOVR_MEAN  |  Mean revenue of voice overage  |
|  VCEOVR_RANGE  |  Range of revenue of voice overage  |
|  Category Variables  |  Explanation  |
|  ACTVSUBS  |  Number of active subscribers in household  |
|  ADULTS  |  Number of adults in household  |
|  AGE1  |  Age of first household member  |
|  AGE2  |  Age of second household member  |
|  AREA  |  Geographic area  |
|  ASL_FLAG  |  Account spending limit  |
|  CAR_BUY  |  New or used car buyer  |
|  CARTYPE  |  Dominant vehicle lifestyle  |
|  CHILDREN  |  Children present in household  |
|  CHURN  |  Instance of churn between 31-60 days after observation date  |
|  CRCLSCOD  |  Credit class code  |
|  CREDITCD  |  Credit card indicator  |
|  CRTCOUNT  |  Adjustments made to credit rating of individual  |
|  CSA  |  Communications local service area  |
|  DIV_TYPE  |  Division type code  |
|  DUALBAND  |  Dualband  |
|  DWLLSIZE  |  Dwelling size  |
|  DWLLTYPE  |  Dwelling unit type  |
|  EDUC1  |  Education of first household member  |
|  ETHNIC  |  Ethnicity roll-up code  |
|  FORGNTVL  |  Foreign travel dummy variable  |
|  HND_PRICE  |  Current handset price  |
|  HHSTATIN  |  Premier household status indicator  |
|  HND_WEBCAP  |  Handset web capability  |
|  INCOME  |  Estimated income  |
|  INFOBASE  |  InfoBase match  |
|  KID0_2  |  Child 0 - 2 years of age in household  |
|  KID3_5  |  Child 3 - 5 years of age in household  |
|  KID6_10  |  Child 6 - 10 years of age in household  |
|  KID11_15  |  Child 11 - 15 years of age in household  |
|  KID16_17  |  Child 16 - 17 years of age in household  |
|  LAST_SWAP  |  Date of last phone swap  |
|  LOR  |  Length of residence  |
|  MAILFLAG  |  DMA: Do not mail flag  |
|  MAILORDR  |  Mail order buyer  |
|  MAILRESP  |  Mail responder  |
|  MARITAL  |  Marital status  |
|  MODELS  |  Number of models issued  |
|  MTRCYCLE  |  Motorcycle indicator  |
|  NEW_CELL  |  New cell phone user  |
|  NUMBCARS  |  Known number of vehicles  |
|  OCCU1  |  Occupation of first household member  |
|  OWNRENT  |  Home owner/renter status  |
|  PCOWNER  |  PC owner dummy variable  |
|  PHONES  |  Number of handsets issued  |
|  PRE_HND_PRICE  |  Previous handset price  |
|  PRIZM_SOCIAL_ONE  |  Social group letter only  |
|  PROPTYPE  |  Property type detail  |
|  REF_QTY  |  Total number of referrals  |
|  REFURB_NEW  |  Handset: refurbished or new  |
|  RV  |  RV indicator  |
|  SOLFLAG  |  Infobase no phone solicitation flag  |
|  TOT_ACPT  |  Total offers accepted from retention team  |
|  TOT_RET  |  Total calls into retention team  |
|  TRUCK  |  Truck indicator  |
|  UNIQSUBS  |  Number of unique subscribers in the household  |
|  WRKWOMAN  |  Working woman in household  |
