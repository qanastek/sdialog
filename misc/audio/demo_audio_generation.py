from sdialog import Dialog, audio

dialog = Dialog.from_file("1.json")

dialog.print()

full_audio = audio.dialog_to_audio(dialog)

audio.to_wav(full_audio, "first_dialog_audio.wav")

dialog.to_audio("first_direct_dialog_audio.wav")

audio_res = dialog.to_audio()
