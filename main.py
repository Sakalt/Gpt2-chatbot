from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# モデルとトークナイザーをロード
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # 評価モードに切り替え

def chat_with_gpt2():
    print("GPT-2と会話を始めましょう！ '終了' と入力すると終了します。")
    
    chat_history = ""  # 会話履歴を保持する

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == '終了':
            break
        
        # 会話履歴にユーザーの入力を追加
        chat_history += f"あなた: {user_input}\n"
        
        # トークン化
        inputs = tokenizer.encode(chat_history, return_tensors='pt')
        
        # モデルによる生成
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
        
        # 生成されたテキストをデコード
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # GPT-2の応答を取得
        bot_response = response.split("\n")[-1]  # 最新の行を取得
        print(f"GPT-2: {bot_response}")

        # 会話履歴にボットの応答を追加
        chat_history += f"GPT-2: {bot_response}\n"

if __name__ == "__main__":
    chat_with_gpt2()
