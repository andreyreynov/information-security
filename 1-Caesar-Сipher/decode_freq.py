from collections import Counter
import frequencies
import matplotlib.pyplot as plt
import numpy as np

message = "Amr11Hifpsex mw e wmqtpi erh pmklxaimklx tsaivwlipp wgvmtx xlex viqsziw tvi-mrwxeppih Amrhsaw fpsexaevi ettw, hmwefpiw xipiqixvc erh higpyxxivw xli ibtivmirgi fc hmwefpmrk sv viqszmrk mrxvywmzi mrxivjegi ipiqirxw, ehw erh qsvi. Rs riih xs temrwxeomrkpc ks xlvsykl epp xli wixxmrkw csyvwipj, sv viqszi ettw sri fc sri. Amr11Hifpsex qeoiw xli tvsgiww uymgo erh iewc! Csy ger tmgo erh glsswi ibegxpc almgl qshmjmgexmsrw csy aerx xli wgvmtx xs qeoi, sv ywi xli hijeypx wixxmrkw. Mj csy evi yrlettc amxl erc sj xli glerkiw csy ger iewmpc vizivx xliq fc ywmrk xli vikmwxvc jmpiw xlex evi mrgpyhih mr xli 'Vikjmpiw' jsphiv, epp sj xli ettw xlex evi viqszih ger fi vimrwxeppih jvsq xli Qmgvswsjx wxsvi."

message = message.upper()


letters = list(frequencies.GetFrequencies(
).get_freqs_percentage(message).items())
letter_1 = letters[0][0]
letter_2 = letters[1][0]
letter_3 = letters[2][0]

print(f"\nMost frequent letter in \033[4m{
      'message'}\033[0m: {letter_1, letter_2, letter_3}")

top_letters = list(frequencies.EnglishFrequecies().letters_sorted.items())
letter_top_1 = top_letters[0][0]
letter_top_2 = top_letters[1][0]
letter_top_3 = top_letters[2][0]

print(f"Most frequent letters in \033[4m{'dictionary'}\033[0m: {
      letter_top_1, letter_top_2, letter_top_3}")

pairs = list(frequencies.GetFrequencies(
).get_digram_percentage(message).items())
pair_1 = pairs[0][0]
pair_2 = pairs[1][0]
pair_3 = pairs[2][0]

print(f"\nMost frequent pairs in \033[4m{
      'message'}\033[0m: {pair_1, pair_2, pair_3}")

top_pairs = list(frequencies.EnglishFrequecies().digram_sorted.items())
pair_top_1 = top_pairs[0][0]
pair_top_2 = top_pairs[1][0]
pair_top_3 = top_pairs[2][0]

print(f"Most frequent pairs in \033[4m{'dictionary'}\033[0m: {
      pair_top_1, pair_top_2, pair_top_3}\n")


def build_plot(data, subplot_num, title):
    x = np.array(data)[:, 0]
    y = np.array(data)[:, 1]

    x = x[::-1]
    y = y[::-1]

    plt.subplot(1, 2, subplot_num)
    plt.bar(x, y)
    plt.xlabel('Syllables')
    plt.ylabel('Frequency, %')
    plt.gca().invert_xaxis()
    plt.title(title)


def show_plot(plot1, title1, plot2, title2):
    # Create a single figure with two subplots
    fig = plt.figure(figsize=(12, 6))

    build_plot(plot1[:10], 1, title1)
    build_plot(plot2[:10], 2, title2)
    plt.show()


show_plot(pairs, 'Top 10 pairs', top_pairs, 'Dictionary Top 10 pairs')
show_plot(letters, 'Top 10 letters', top_letters, 'Dictionary Top 10 letters')
