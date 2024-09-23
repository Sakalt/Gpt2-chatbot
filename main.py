from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# モデルとトークナイザーをロード
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def chat_with_gpt2():
    print("GPT-2と会話を始めましょう！ '終了' と入力すると終了します。")
    
    chat_history = ""

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == '終了':
            break
        
        chat_history += f"あなた: {user_input}\n"
        
        inputs = tokenizer.encode(chat_history, return_tensors='pt')
        
        # 生成時の設定を調整
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=200, num_return_sequences=1, temperature=0.8, top_k=50, top_p=0.95)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = response.split("\n")[-1]
        print(f"GPT-2: {bot_response}")

        chat_history += f"GPT-2: {bot_response}\n"

if __name__ == "__main__":
    chat_with_gpt2()
