def generate_maqam_frequencies(maqam_name, num_of_notes, root_freq, step_size=1):
    """Generate a list of frequencies based on Arabic Maqamat tuning with quarter-tones, extending to additional octaves if needed."""
    
    # Define Maqamat with microtonal (quarter-tone) tuning ratios
    maqamat = {
        "Rast": [1, 9/8, 11/9, 4/3, 3/2, 5/3, 15/8, 2],  # Quarter-tone 3rd & 7th degrees
        "Bayati": [1, 17/16, 6/5, 4/3, 3/2, 8/5, 9/5, 2],  # Quarter-tone 2nd & 3rd degrees
        "Saba": [1, 17/16, 6/5, 7/5, 3/2, 8/5, 15/8, 2],  # Quarter-tone 2nd, 3rd & 6th degrees
        "Hijaz": [1, 16/15, 6/5, 14/9, 3/2, 8/5, 15/8, 2],  # Quarter-tone 3rd degree
        "Nahawand": [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2],  # Natural minor equivalent in Just Intonation
        "Kurd": [1, 17/16, 6/5, 4/3, 3/2, 8/5, 9/5, 2],  # Quarter-tone 2nd degree, Phrygian-like
        "Nahawand Murassah": [1, 17/16, 6/5, 4/3, 3/2, 8/5, 15/8, 2],  # Nahawand with quarter-tone 2nd degree
        "Sikah": [1, 13/12, 6/5, 4/3, 3/2, 8/5, 9/5, 2]  # Historically accurate, with a neutral second
    }
    
    if maqam_name not in maqamat:
        raise ValueError("Invalid Maqam name. Choose from: Rast, Bayati, Saba, Hijaz, Nahawand")
    
    # Get the frequency ratios for the selected Maqam
    maqam_ratios = maqamat[maqam_name]
    
    # Extend maqam by repeating it in higher octaves if needed
    extended_ratios = []
    octave = 1
    while len(extended_ratios) < num_of_notes*step_size:
        extended_ratios.extend([r * octave for r in maqam_ratios])
        octave *= 2  # Move to the next octave
    
    # Apply step size selection to spread out notes
    selected_ratios = [extended_ratios[i] for i in range(0, len(extended_ratios), step_size)][:num_of_notes]
    
    maqam_frequencies=[]
    # Generate frequencies by multiplying the root frequency with the selected ratios
    for tone in [root_freq * ratio for ratio in selected_ratios]:
        if tone < 800:
            maqam_frequencies.append(tone)
        else:
            maqam_frequencies.append(tone/2)

    
    return maqam_frequencies

# Example usage
# maqam_freqs = generate_maqam_frequencies("Sikah", 12, 70, step_size=3)
# print(maqam_freqs)