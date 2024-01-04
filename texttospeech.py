from gtts import gTTS
import os

def text_to_speech(text):
    # Simpan teks ke file sementara
    tts = gTTS(text=text, lang='id')
    tts.save("temp.mp3")

    # Putar file audio menggunakan pemutar default
    os.system("start temp.mp3")

def main():
    current_text = ""

    while True:
        user_input = input("Masukkan karakter atau angka 1 untuk mendengarkan kalimat: ")

        if user_input == "1":
            if current_text:
                print("Mengucapkan kalimat:", current_text)
                text_to_speech(current_text)
                current_text = ""
            else:
                print("Tidak ada kalimat untuk diucapkan.")
        else:
            current_text += user_input

if __name__ == "__main__":
    main()
