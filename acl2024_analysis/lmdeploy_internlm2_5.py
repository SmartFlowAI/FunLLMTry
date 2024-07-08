from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(
        rope_scaling_factor=2.5,
        session_len=1000000,
        max_batch_size=1,
        cache_max_entry_count=0.7,
        tp=4)
pipe = pipeline('internlm/internlm2_5-7b-chat-1m', backend_config=backend_config)

data = open('ac_papers.txt', encoding='utf-8').read()
prompt = f'{data}\n这是2024年 ACL 录取论文列表，请从多个角度总结出有哪些特点'
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
response = pipe(prompt, gen_config=gen_config)
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(response.text)
print('Done! Please check the output.txt file for the result.')
