

class Config:
    def __init__(self):

        # generate data parm
        self.prompt_vol = 3 # examples using each round
        self.total_data_gene = 4000
        self.temperature = 0.9
        self.rep_penalty = 1.1
        # file name
        self.few_shot_idx = 7
        self.prompt_file = f"../restaurant8k/train_{self.few_shot_idx}.json"
        self.check_point_file = f'check_syn_few{self.few_shot_idx}_0603_1_4000.json'
        self.outputFile = f'syn_few{self.few_shot_idx}_0603_1_4000.json'

        self.use_all_data = True
        self.use_data_num = 15
        # if want to use hugging face model can type model name else None
        self.hugging_face_llm = None

        self.slot_ask = ['people', 'date', 'time', 'first_name', 'last_name']
        self.slot_ask_part = [4, 4, 4, 2, 2]
        self.slot_ask_num = [int((ask_part / sum(self.slot_ask_part)) * self.total_data_gene * 1.3) for ask_part in self.slot_ask_part]

        self.prefix_1 = "Generate examples include some of these slot: "
        self.prefix_2 = ". Here is an example sentence: "
        self.default_prefix = self.prefix_1 + str(self.slot_ask) + self.prefix_2
        self.suffix = f"""Example{self.prompt_vol + 1}: """
