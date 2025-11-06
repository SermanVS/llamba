import pandas as pd
import torch

from llamba.chatmodels.chat_model import AbstractChatModel
from llamba.util.disease import Disease
from llamba_library.bioage_model import BioAgeModel

class LlambaConnector:
    def __init__(self, bioage_model: BioAgeModel, chat_model: AbstractChatModel):
        self.bioage_model = bioage_model
        self.chat_model = chat_model
        self.answer = ""
        self.clock = ""

    def specify_clock(self, clock: str):
        self.clock = clock

    def produce_risk_prompt(self, disease: Disease):
        disease_prompt = ""
        
        if self.clock:
            disease_prompt += f"Use {self.clock} for a reference as an epigenetic clock. "

        disease_prompt += f"Given the following parameters, estimate the risk of {disease.name} occurring: "
        for i in self.top_n:
            disease_prompt += f'{self.top_shap['feats'][i]}: {self.top_shap['data'][i]}\n'
            
        return disease_prompt

    def produce_basic_answer(self):
        self.answer += 'Your bioage is {bio_age} and your aging acceleration is {acceleration}, which means ' \
            .format(bio_age=round(self.bio_age), 
                    acceleration=round(self.acceleration[0]))

        if (self.acceleration > 1):
            self.answer += 'you are ageing quicker than normal.\n\n'
        elif (self.acceleration > -1 and self.acceleration < 1):
            self.answer += 'you are ageing normally.\n\n'
        else:
            self.answer += 'you are ageing slower than normal.\n\n'

    def produce_advanced_answer(self):
        self.answer += 'Here is some more information about your data. \n\n'
        self.query_prompts()

    def analyze(self, data: pd.DataFrame, device=torch.device('cpu'), analyze_feats=False, analyze_risks=False, **kwargs):
        self._analyze_feats = analyze_feats
        self._analyze_risks = analyze_risks

        data['bio_age'] = self.bioage_model.inference(data=data.drop(['Age'], axis=1), device=device)
        self.bio_age = data['bio_age'].values[0]
        self.acceleration = data['bio_age'].values - data['Age'].values
        self.shap_dict = kwargs.get('shap_dict', None)
        self.top_n = kwargs.get('top_n', None)

        # If we want to get info about features like what they mean and what their increased/decreased levels mean
        if analyze_feats:
            if self.shap_dict and self.top_n:
                return self.advanced_analysis()

        self.produce_basic_answer()
        return {"analysis": self.answer, "acceleration": self.acceleration[0]}
    
    def advanced_analysis(self, data: pd.DataFrame):
        self.feats = data.drop(['Age', 'bio_age'], axis=1).columns.to_list()
        self.top_shap = self.bioage_model.get_top_shap(self.top_n, data, self.feats, self.shap_dict)
        
        feat_prompts = self.produce_feat_analysis_prompts(top_n=self.top_n, data=self.top_shap['data'], feats=self.top_shap['feats'], values=self.top_shap['values'])
        self.produce_basic_answer()
        self.produce_advanced_answer(feat_prompts)
        return {"analysis": self.answer, "acceleration": self.acceleration[0], "features": self.feats}    

    def produce_feat_analysis_prompts(self, top_n, data, feats, values):
        prompts = []
        for i in range(top_n):
            self.answer += f'{feats[i]}: {data[i]}\n'
            if values[i] > 0:
                level = 'an increased'
            else:
                level = 'a reduced'
            prompts.append(f'What is {feats[i]}? What does {level} level of {feats[i]} mean?')
        return prompts
    
    def query_prompt(self, prompt: str):
        res = self.chat_model.query(prompt=prompt)[1]
        return res

    def query_prompts(self, prompts: list):
        for prompt in prompts:
            res = self.query_prompt(prompt)
            self.answer += res
            self.answer += '\n\n'
        return self.answer