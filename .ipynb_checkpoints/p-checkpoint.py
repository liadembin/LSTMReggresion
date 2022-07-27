{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b3a60d0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np \n",
    "import tensorflow as tf \n",
    "from tensorflow import keras \n",
    "from tensorflow.keras.layers import GRU,LSTM,Dense,Flatten,Dropout\n",
    "from tensorflow.keras.models import Sequential\n",
    "from matplotlib import pyplot as plt \n",
    "from tensorflow.keras.optimizers import Adam,RMSprop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d668b91c",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"./data/data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6b2ef66b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Adj Close</th>\n",
       "      <th>MFV</th>\n",
       "      <th>40 period ADX.</th>\n",
       "      <th>AO</th>\n",
       "      <th>UPPER</th>\n",
       "      <th>LOWER</th>\n",
       "      <th>40 period ATR</th>\n",
       "      <th>Buy.</th>\n",
       "      <th>Sell.</th>\n",
       "      <th>...</th>\n",
       "      <th>SIGNAL.3</th>\n",
       "      <th>VZO</th>\n",
       "      <th>40 Williams %R</th>\n",
       "      <th>40 period WMA.</th>\n",
       "      <th>WOBV</th>\n",
       "      <th>WT1.</th>\n",
       "      <th>WT2.</th>\n",
       "      <th>40 period ZLEMA</th>\n",
       "      <th>pct</th>\n",
       "      <th>pct_shifted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2019-09-20 09:30:00-04:00</td>\n",
       "      <td>16.407499</td>\n",
       "      <td>2.116714e+07</td>\n",
       "      <td>14.309929</td>\n",
       "      <td>0.182759</td>\n",
       "      <td>16.689889</td>\n",
       "      <td>16.054189</td>\n",
       "      <td>0.153630</td>\n",
       "      <td>0.497433</td>\n",
       "      <td>1.605319</td>\n",
       "      <td>...</td>\n",
       "      <td>0.039537</td>\n",
       "      <td>12.117903</td>\n",
       "      <td>-33.487389</td>\n",
       "      <td>16.251470</td>\n",
       "      <td>9.670663e+06</td>\n",
       "      <td>20.127102</td>\n",
       "      <td>18.448995</td>\n",
       "      <td>16.353788</td>\n",
       "      <td>0.034847</td>\n",
       "      <td>0.027954</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2019-09-20 10:30:00-04:00</td>\n",
       "      <td>16.260000</td>\n",
       "      <td>1.943094e+07</td>\n",
       "      <td>14.140038</td>\n",
       "      <td>0.171225</td>\n",
       "      <td>16.685534</td>\n",
       "      <td>16.049697</td>\n",
       "      <td>0.154255</td>\n",
       "      <td>0.114767</td>\n",
       "      <td>1.950409</td>\n",
       "      <td>...</td>\n",
       "      <td>0.041372</td>\n",
       "      <td>7.085278</td>\n",
       "      <td>-45.734591</td>\n",
       "      <td>16.251063</td>\n",
       "      <td>9.382564e+06</td>\n",
       "      <td>15.500417</td>\n",
       "      <td>18.108371</td>\n",
       "      <td>16.356737</td>\n",
       "      <td>0.031090</td>\n",
       "      <td>0.034847</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2019-09-20 11:30:00-04:00</td>\n",
       "      <td>16.194000</td>\n",
       "      <td>1.931478e+07</td>\n",
       "      <td>14.107290</td>\n",
       "      <td>0.141156</td>\n",
       "      <td>16.674920</td>\n",
       "      <td>16.039176</td>\n",
       "      <td>0.155231</td>\n",
       "      <td>0.710949</td>\n",
       "      <td>0.801172</td>\n",
       "      <td>...</td>\n",
       "      <td>0.040254</td>\n",
       "      <td>2.478898</td>\n",
       "      <td>-51.990512</td>\n",
       "      <td>16.247949</td>\n",
       "      <td>9.259769e+06</td>\n",
       "      <td>6.009721</td>\n",
       "      <td>14.930795</td>\n",
       "      <td>16.351900</td>\n",
       "      <td>0.018428</td>\n",
       "      <td>0.031090</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 119 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Unnamed: 0  Adj Close           MFV  40 period ADX.  \\\n",
       "0  2019-09-20 09:30:00-04:00  16.407499  2.116714e+07       14.309929   \n",
       "1  2019-09-20 10:30:00-04:00  16.260000  1.943094e+07       14.140038   \n",
       "2  2019-09-20 11:30:00-04:00  16.194000  1.931478e+07       14.107290   \n",
       "\n",
       "         AO      UPPER      LOWER  40 period ATR      Buy.     Sell.  ...  \\\n",
       "0  0.182759  16.689889  16.054189       0.153630  0.497433  1.605319  ...   \n",
       "1  0.171225  16.685534  16.049697       0.154255  0.114767  1.950409  ...   \n",
       "2  0.141156  16.674920  16.039176       0.155231  0.710949  0.801172  ...   \n",
       "\n",
       "   SIGNAL.3        VZO  40 Williams %R  40 period WMA.          WOBV  \\\n",
       "0  0.039537  12.117903      -33.487389       16.251470  9.670663e+06   \n",
       "1  0.041372   7.085278      -45.734591       16.251063  9.382564e+06   \n",
       "2  0.040254   2.478898      -51.990512       16.247949  9.259769e+06   \n",
       "\n",
       "        WT1.       WT2.  40 period ZLEMA       pct  pct_shifted  \n",
       "0  20.127102  18.448995        16.353788  0.034847     0.027954  \n",
       "1  15.500417  18.108371        16.356737  0.031090     0.034847  \n",
       "2   6.009721  14.930795        16.351900  0.018428     0.031090  \n",
       "\n",
       "[3 rows x 119 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7a951c9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pandas_profiling import ProfileReport"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "abfb4f22",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data.drop(\"Unnamed: 0\",axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5cb59580",
   "metadata": {},
   "outputs": [],
   "source": [
    "prof = ProfileReport(data,title=\"a2\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e453888d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7415f20a01104fdf98682f1f27fd0654",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Input \u001b[1;32mIn [7]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mprof\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mto_file\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mprofile.html\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\profile_report.py:257\u001b[0m, in \u001b[0;36mProfileReport.to_file\u001b[1;34m(self, output_file, silent)\u001b[0m\n\u001b[0;32m    254\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39mhtml\u001b[38;5;241m.\u001b[39massets_prefix \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mstr\u001b[39m(output_file\u001b[38;5;241m.\u001b[39mstem) \u001b[38;5;241m+\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m_assets\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    255\u001b[0m     create_html_assets(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig, output_file)\n\u001b[1;32m--> 257\u001b[0m data \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mto_html\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    259\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m output_file\u001b[38;5;241m.\u001b[39msuffix \u001b[38;5;241m!=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m.html\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m    260\u001b[0m     suffix \u001b[38;5;241m=\u001b[39m output_file\u001b[38;5;241m.\u001b[39msuffix\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\profile_report.py:368\u001b[0m, in \u001b[0;36mProfileReport.to_html\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    360\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mto_html\u001b[39m(\u001b[38;5;28mself\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28mstr\u001b[39m:\n\u001b[0;32m    361\u001b[0m     \u001b[38;5;124;03m\"\"\"Generate and return complete template as lengthy string\u001b[39;00m\n\u001b[0;32m    362\u001b[0m \u001b[38;5;124;03m        for using with frameworks.\u001b[39;00m\n\u001b[0;32m    363\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    366\u001b[0m \n\u001b[0;32m    367\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 368\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mhtml\u001b[49m\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\profile_report.py:185\u001b[0m, in \u001b[0;36mProfileReport.html\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    182\u001b[0m \u001b[38;5;129m@property\u001b[39m\n\u001b[0;32m    183\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mhtml\u001b[39m(\u001b[38;5;28mself\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28mstr\u001b[39m:\n\u001b[0;32m    184\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_html \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 185\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_html \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_render_html\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    186\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_html\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\profile_report.py:287\u001b[0m, in \u001b[0;36mProfileReport._render_html\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    284\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_render_html\u001b[39m(\u001b[38;5;28mself\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28mstr\u001b[39m:\n\u001b[0;32m    285\u001b[0m     \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mpandas_profiling\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mreport\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpresentation\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mflavours\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m HTMLReport\n\u001b[1;32m--> 287\u001b[0m     report \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mreport\u001b[49m\n\u001b[0;32m    289\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m tqdm(\n\u001b[0;32m    290\u001b[0m         total\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m, desc\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mRender HTML\u001b[39m\u001b[38;5;124m\"\u001b[39m, disable\u001b[38;5;241m=\u001b[39m\u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39mprogress_bar\n\u001b[0;32m    291\u001b[0m     ) \u001b[38;5;28;01mas\u001b[39;00m pbar:\n\u001b[0;32m    292\u001b[0m         html \u001b[38;5;241m=\u001b[39m HTMLReport(copy\u001b[38;5;241m.\u001b[39mdeepcopy(report))\u001b[38;5;241m.\u001b[39mrender(\n\u001b[0;32m    293\u001b[0m             nav\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39mhtml\u001b[38;5;241m.\u001b[39mnavbar_show,\n\u001b[0;32m    294\u001b[0m             offline\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39mhtml\u001b[38;5;241m.\u001b[39muse_local_assets,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    302\u001b[0m             version\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdescription_set[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpackage\u001b[39m\u001b[38;5;124m\"\u001b[39m][\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpandas_profiling_version\u001b[39m\u001b[38;5;124m\"\u001b[39m],\n\u001b[0;32m    303\u001b[0m         )\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\profile_report.py:179\u001b[0m, in \u001b[0;36mProfileReport.report\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    176\u001b[0m \u001b[38;5;129m@property\u001b[39m\n\u001b[0;32m    177\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mreport\u001b[39m(\u001b[38;5;28mself\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Root:\n\u001b[0;32m    178\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_report \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 179\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_report \u001b[38;5;241m=\u001b[39m get_report_structure(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig, \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdescription_set\u001b[49m)\n\u001b[0;32m    180\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_report\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\profile_report.py:161\u001b[0m, in \u001b[0;36mProfileReport.description_set\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    158\u001b[0m \u001b[38;5;129m@property\u001b[39m\n\u001b[0;32m    159\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mdescription_set\u001b[39m(\u001b[38;5;28mself\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Dict[\u001b[38;5;28mstr\u001b[39m, Any]:\n\u001b[0;32m    160\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_description_set \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 161\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_description_set \u001b[38;5;241m=\u001b[39m \u001b[43mdescribe_df\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    162\u001b[0m \u001b[43m            \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mconfig\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    163\u001b[0m \u001b[43m            \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdf\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    164\u001b[0m \u001b[43m            \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msummarizer\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    165\u001b[0m \u001b[43m            \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtypeset\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    166\u001b[0m \u001b[43m            \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_sample\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    167\u001b[0m \u001b[43m        \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    168\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_description_set\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\model\\describe.py:95\u001b[0m, in \u001b[0;36mdescribe\u001b[1;34m(config, df, summarizer, typeset, sample)\u001b[0m\n\u001b[0;32m     92\u001b[0m correlation_names \u001b[38;5;241m=\u001b[39m get_active_correlations(config)\n\u001b[0;32m     93\u001b[0m pbar\u001b[38;5;241m.\u001b[39mtotal \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(correlation_names)\n\u001b[1;32m---> 95\u001b[0m correlations \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m     96\u001b[0m     correlation_name: progress(\n\u001b[0;32m     97\u001b[0m         calculate_correlation, pbar, \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCalculate \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mcorrelation_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m correlation\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     98\u001b[0m     )(config, df, correlation_name, series_description)\n\u001b[0;32m     99\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m correlation_name \u001b[38;5;129;01min\u001b[39;00m correlation_names\n\u001b[0;32m    100\u001b[0m }\n\u001b[0;32m    102\u001b[0m \u001b[38;5;66;03m# make sure correlations is not None\u001b[39;00m\n\u001b[0;32m    103\u001b[0m correlations \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m    104\u001b[0m     key: value \u001b[38;5;28;01mfor\u001b[39;00m key, value \u001b[38;5;129;01min\u001b[39;00m correlations\u001b[38;5;241m.\u001b[39mitems() \u001b[38;5;28;01mif\u001b[39;00m value \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    105\u001b[0m }\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\model\\describe.py:96\u001b[0m, in \u001b[0;36m<dictcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m     92\u001b[0m correlation_names \u001b[38;5;241m=\u001b[39m get_active_correlations(config)\n\u001b[0;32m     93\u001b[0m pbar\u001b[38;5;241m.\u001b[39mtotal \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(correlation_names)\n\u001b[0;32m     95\u001b[0m correlations \u001b[38;5;241m=\u001b[39m {\n\u001b[1;32m---> 96\u001b[0m     correlation_name: \u001b[43mprogress\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m     97\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcalculate_correlation\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mpbar\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43mf\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mCalculate \u001b[39;49m\u001b[38;5;132;43;01m{\u001b[39;49;00m\u001b[43mcorrelation_name\u001b[49m\u001b[38;5;132;43;01m}\u001b[39;49;00m\u001b[38;5;124;43m correlation\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\n\u001b[0;32m     98\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\u001b[43m(\u001b[49m\u001b[43mconfig\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdf\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcorrelation_name\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mseries_description\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     99\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m correlation_name \u001b[38;5;129;01min\u001b[39;00m correlation_names\n\u001b[0;32m    100\u001b[0m }\n\u001b[0;32m    102\u001b[0m \u001b[38;5;66;03m# make sure correlations is not None\u001b[39;00m\n\u001b[0;32m    103\u001b[0m correlations \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m    104\u001b[0m     key: value \u001b[38;5;28;01mfor\u001b[39;00m key, value \u001b[38;5;129;01min\u001b[39;00m correlations\u001b[38;5;241m.\u001b[39mitems() \u001b[38;5;28;01mif\u001b[39;00m value \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    105\u001b[0m }\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\utils\\progress_bar.py:11\u001b[0m, in \u001b[0;36mprogress.<locals>.inner\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[38;5;129m@wraps\u001b[39m(fn)\n\u001b[0;32m      9\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21minner\u001b[39m(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Any:\n\u001b[0;32m     10\u001b[0m     bar\u001b[38;5;241m.\u001b[39mset_postfix_str(message)\n\u001b[1;32m---> 11\u001b[0m     ret \u001b[38;5;241m=\u001b[39m fn(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m     12\u001b[0m     bar\u001b[38;5;241m.\u001b[39mupdate()\n\u001b[0;32m     13\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m ret\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\model\\correlations.py:94\u001b[0m, in \u001b[0;36mcalculate_correlation\u001b[1;34m(config, df, correlation_name, summary)\u001b[0m\n\u001b[0;32m     92\u001b[0m correlation \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m     93\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m---> 94\u001b[0m     correlation \u001b[38;5;241m=\u001b[39m \u001b[43mcorrelation_measures\u001b[49m\u001b[43m[\u001b[49m\u001b[43mcorrelation_name\u001b[49m\u001b[43m]\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcompute\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m     95\u001b[0m \u001b[43m        \u001b[49m\u001b[43mconfig\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdf\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43msummary\u001b[49m\n\u001b[0;32m     96\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     97\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m (\u001b[38;5;167;01mValueError\u001b[39;00m, \u001b[38;5;167;01mAssertionError\u001b[39;00m, \u001b[38;5;167;01mTypeError\u001b[39;00m, DataError, \u001b[38;5;167;01mIndexError\u001b[39;00m) \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m     98\u001b[0m     warn_correlation(correlation_name, \u001b[38;5;28mstr\u001b[39m(e))\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\multimethod\\__init__.py:312\u001b[0m, in \u001b[0;36mmultimethod.__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    310\u001b[0m func \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m[\u001b[38;5;28mtuple\u001b[39m(func(arg) \u001b[38;5;28;01mfor\u001b[39;00m func, arg \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mzip\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtype_checkers, args))]\n\u001b[0;32m    311\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 312\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    313\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m ex:\n\u001b[0;32m    314\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m DispatchError(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFunction \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfunc\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__code__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mex\u001b[39;00m\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\pandas_profiling\\model\\pandas\\correlations_pandas.py:134\u001b[0m, in \u001b[0;36mpandas_phik_compute\u001b[1;34m(config, df, summary)\u001b[0m\n\u001b[0;32m    131\u001b[0m     warnings\u001b[38;5;241m.\u001b[39msimplefilter(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mignore\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m    132\u001b[0m     \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mphik\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m phik_matrix\n\u001b[1;32m--> 134\u001b[0m     correlation \u001b[38;5;241m=\u001b[39m \u001b[43mphik_matrix\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdf\u001b[49m\u001b[43m[\u001b[49m\u001b[43mselected_cols\u001b[49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43minterval_cols\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mlist\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mintcols\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    136\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m correlation\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\phik\\phik.py:256\u001b[0m, in \u001b[0;36mphik_matrix\u001b[1;34m(df, interval_cols, bins, quantile, noise_correction, dropna, drop_underflow, drop_overflow, verbose, njobs)\u001b[0m\n\u001b[0;32m    223\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    224\u001b[0m \u001b[38;5;124;03mCorrelation matrix of bivariate gaussian derived from chi2-value\u001b[39;00m\n\u001b[0;32m    225\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    245\u001b[0m \u001b[38;5;124;03m:return: phik correlation matrix\u001b[39;00m\n\u001b[0;32m    246\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    248\u001b[0m data_binned, binning_dict \u001b[38;5;241m=\u001b[39m auto_bin_data(\n\u001b[0;32m    249\u001b[0m     df\u001b[38;5;241m=\u001b[39mdf,\n\u001b[0;32m    250\u001b[0m     interval_cols\u001b[38;5;241m=\u001b[39minterval_cols,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    254\u001b[0m     verbose\u001b[38;5;241m=\u001b[39mverbose,\n\u001b[0;32m    255\u001b[0m )\n\u001b[1;32m--> 256\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mphik_from_rebinned_df\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    257\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdata_binned\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    258\u001b[0m \u001b[43m    \u001b[49m\u001b[43mnoise_correction\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    259\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdropna\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdropna\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    260\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdrop_underflow\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdrop_underflow\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    261\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdrop_overflow\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdrop_overflow\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    262\u001b[0m \u001b[43m    \u001b[49m\u001b[43mnjobs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mnjobs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    263\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\phik\\phik.py:166\u001b[0m, in \u001b[0;36mphik_from_rebinned_df\u001b[1;34m(data_binned, noise_correction, dropna, drop_underflow, drop_overflow, njobs)\u001b[0m\n\u001b[0;32m    159\u001b[0m     phik_list \u001b[38;5;241m=\u001b[39m [\n\u001b[0;32m    160\u001b[0m         _calc_phik(co, data_binned[\u001b[38;5;28mlist\u001b[39m(co)], noise_correction)\n\u001b[0;32m    161\u001b[0m         \u001b[38;5;28;01mfor\u001b[39;00m co \u001b[38;5;129;01min\u001b[39;00m itertools\u001b[38;5;241m.\u001b[39mcombinations_with_replacement(\n\u001b[0;32m    162\u001b[0m             data_binned\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mvalues, \u001b[38;5;241m2\u001b[39m\n\u001b[0;32m    163\u001b[0m         )\n\u001b[0;32m    164\u001b[0m     ]\n\u001b[0;32m    165\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 166\u001b[0m     phik_list \u001b[38;5;241m=\u001b[39m \u001b[43mParallel\u001b[49m\u001b[43m(\u001b[49m\u001b[43mn_jobs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mnjobs\u001b[49m\u001b[43m)\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    167\u001b[0m \u001b[43m        \u001b[49m\u001b[43mdelayed\u001b[49m\u001b[43m(\u001b[49m\u001b[43m_calc_phik\u001b[49m\u001b[43m)\u001b[49m\u001b[43m(\u001b[49m\u001b[43mco\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdata_binned\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;28;43mlist\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mco\u001b[49m\u001b[43m)\u001b[49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnoise_correction\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    168\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mco\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mitertools\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcombinations_with_replacement\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    169\u001b[0m \u001b[43m            \u001b[49m\u001b[43mdata_binned\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcolumns\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mvalues\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m2\u001b[39;49m\n\u001b[0;32m    170\u001b[0m \u001b[43m        \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    171\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    173\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(phik_list) \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m    174\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m pd\u001b[38;5;241m.\u001b[39mDataFrame(np\u001b[38;5;241m.\u001b[39mnan, index\u001b[38;5;241m=\u001b[39mcolumn_order, columns\u001b[38;5;241m=\u001b[39mcolumn_order)\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\joblib\\parallel.py:1056\u001b[0m, in \u001b[0;36mParallel.__call__\u001b[1;34m(self, iterable)\u001b[0m\n\u001b[0;32m   1053\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_iterating \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mFalse\u001b[39;00m\n\u001b[0;32m   1055\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backend\u001b[38;5;241m.\u001b[39mretrieval_context():\n\u001b[1;32m-> 1056\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mretrieve\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1057\u001b[0m \u001b[38;5;66;03m# Make sure that we get a last message telling us we are done\u001b[39;00m\n\u001b[0;32m   1058\u001b[0m elapsed_time \u001b[38;5;241m=\u001b[39m time\u001b[38;5;241m.\u001b[39mtime() \u001b[38;5;241m-\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_start_time\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\joblib\\parallel.py:935\u001b[0m, in \u001b[0;36mParallel.retrieve\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    933\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m    934\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mgetattr\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backend, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124msupports_timeout\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;28;01mFalse\u001b[39;00m):\n\u001b[1;32m--> 935\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_output\u001b[38;5;241m.\u001b[39mextend(\u001b[43mjob\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtimeout\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtimeout\u001b[49m\u001b[43m)\u001b[49m)\n\u001b[0;32m    936\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    937\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_output\u001b[38;5;241m.\u001b[39mextend(job\u001b[38;5;241m.\u001b[39mget())\n",
      "File \u001b[1;32mC:\\Python310\\lib\\site-packages\\joblib\\_parallel_backends.py:542\u001b[0m, in \u001b[0;36mLokyBackend.wrap_future_result\u001b[1;34m(future, timeout)\u001b[0m\n\u001b[0;32m    539\u001b[0m \u001b[38;5;124;03m\"\"\"Wrapper for Future.result to implement the same behaviour as\u001b[39;00m\n\u001b[0;32m    540\u001b[0m \u001b[38;5;124;03mAsyncResults.get from multiprocessing.\"\"\"\u001b[39;00m\n\u001b[0;32m    541\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 542\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfuture\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mresult\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtimeout\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtimeout\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    543\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m CfTimeoutError \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m    544\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTimeoutError\u001b[39;00m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01me\u001b[39;00m\n",
      "File \u001b[1;32mC:\\Python310\\lib\\concurrent\\futures\\_base.py:441\u001b[0m, in \u001b[0;36mFuture.result\u001b[1;34m(self, timeout)\u001b[0m\n\u001b[0;32m    438\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_state \u001b[38;5;241m==\u001b[39m FINISHED:\n\u001b[0;32m    439\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m__get_result()\n\u001b[1;32m--> 441\u001b[0m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_condition\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mwait\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtimeout\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    443\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_state \u001b[38;5;129;01min\u001b[39;00m [CANCELLED, CANCELLED_AND_NOTIFIED]:\n\u001b[0;32m    444\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m CancelledError()\n",
      "File \u001b[1;32mC:\\Python310\\lib\\threading.py:320\u001b[0m, in \u001b[0;36mCondition.wait\u001b[1;34m(self, timeout)\u001b[0m\n\u001b[0;32m    318\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:    \u001b[38;5;66;03m# restore state no matter what (e.g., KeyboardInterrupt)\u001b[39;00m\n\u001b[0;32m    319\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m timeout \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 320\u001b[0m         \u001b[43mwaiter\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43macquire\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    321\u001b[0m         gotit \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[0;32m    322\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "prof.to_file(\"profile.html\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "be9220b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Unnamed: 0', 'Adj Close', 'MFV', '40 period ADX.', 'AO', 'UPPER',\n",
       "       'LOWER', '40 period ATR', 'Buy.', 'Sell.',\n",
       "       ...\n",
       "       'SIGNAL.3', 'VZO', '40 Williams %R', '40 period WMA.', 'WOBV', 'WT1.',\n",
       "       'WT2.', '40 period ZLEMA', 'pct', 'pct_shifted'],\n",
       "      dtype='object', length=119)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2129c8be",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
