from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# モデルとトークナイザーをロード
model_name = 'Sakalti/kanide'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # 評価モードに切り替え

def chat_with_kanide():
    print("Kanideと会話を始めましょう！ '終了' と入力すると終了します。")
    
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
            outputs = model.generate(inputs, max_length=200, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95)
        
        # 生成されたテキストをデコード
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Kanideの応答を取得
        bot_response = response.split("\n")[-1]  # 最新の行を取得
        print(f"Kanide: {bot_response}")

        # 会話履歴にボットの応答を追加
        chat_history += f"Kanide: {bot_response}\n"

if __name__ == "__main__":
    chat_with_kanide()
