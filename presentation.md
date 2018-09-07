---
title:  "Binaural processing in normal and hearing impaired listeners"
author: "Jaime A. Undurraga"
date: "2018-08-16"
output: 
 revealjs::revealjs_presentation:
   theme: white
   transition: fade
   slide_level: 2
   self_contained: false
   reveal_plugins: ["zoom", "notes", "chalkboard"]
   reveal_options:
     chalkboard: 
       toggleNotesButton: true
     slideNumber: true
     progress: true
     previewLinks: true
   includes:
     in_header: header.html
     before_body: doc_prefix.html
     after_body: doc_suffix.html
   css: presentation.css
   # pandoc_args: [ "--output=index.html"] 
# bibliography: ./library.bib  
---



## Before starting 

Binaural audio examples need to be heard using headphones (both, left and rigth!!).
You can get this presentation and play the examples at

https://jundurraga.github.io/MAPS2018/#/

<img src="./my_figures/qrcode.png" width="20%">

A printable version of this presentation can be obtained at

https://jundurraga.github.io/MAPS2018/?print-pdf#/

# Introduction

## Cocktail-party problem (Cherry, 1953)

How do we recognize what one person is saying when others are speaking at the same time?

When everyone at a well-attended party talks at the same level, the speech of the attended talker at a distance of 0.7 m has a signal-to-noise ratio (SNR) of about 0 dB (the background is as intense as the target talker Plomp 1977). 
This level is sufficient to give adequate intelligibility for listeners with normal hearing (Miller, 1947).


<img src="./my_figures/cocktail_party_effect.png" width="50%">

## Auditory scene analysis
- A critial role of the auditory system is to parse the inputs from the left and right ears into auditory objects - auditory scene analysis. This is an essential role for survival, recognition, and communication.

- Binaural hearing provides cues that allow us to estimate the relative number and location of sources and objects in the environment.

- These cues also help us to estimate the dimensions and characteristics of rooms as well as to hear out speakers in the presence of interfering noise.


<img src="./my_figures/casa_Grothe_2010.png" width="50%">

Grothe et al. (2010)

## 

>- When the speech sources are spatially separated, normal listeners can perceptually segregate and selectively attend to the
source of interest if the sounds arrive from the same location - **spatial release from masking (SRM) ** (Freyman et al., 1999; Brungart, 2001; Freyman
et al., 2001; Hawley et al., 2004)


>- This task is extreamly challenging for listeners with sensorineural hearing loss (with or without hearing aids) or cochlear implant users (Loizou et al., 2009; Marrone et al., 2008).


>- Moreover, listeners with "normal" hearing and elder listeners experience great difficulty when listening in the presence of background noise (Ruggles et al., 2011; Swaminathan et al., 2015; Gallun et al., 2013).

>- Hearing impaired listeners with symmetric binaural hearing often demonstrate reduced SRM primarily due to increased thresholds in spatially separated conditions (Arbogast et al., 2005; Marrone et al., 2008b; Best et al., 2012)

# Monaural and binaural cues

## Monaural 

## 

<img class="fragment" data-fragment-index="0" src="./my_figures/Anatomy_of_the_Human_Ear.svg" width="45%">
<img class="fragment" data-fragment-index="1" src="./my_figures/Uncoiled_cochlea_with_basilar_membrane_color.png" width="45%">
<img class="fragment" data-fragment-index="2" src="./my_figures/auditory_nerve.png" width="25%">

## Monaural cues and Auditory scence analysis

- Normal hearing listeners can segregate sounds remarkably well from monaural signals.
- The brain can use the statistical regularities of a sound to group acoustic characteristics that are likely originated from the same source.

<div class="column" style="float:left; width:20%; text-align: left">
<br>
<br>
<br>
<li class="fragment" data-fragment-index="0"> common onset </li> 
<br>
<li class="fragment" data-fragment-index="1"> harmonicity </li> 
<br>
<li class="fragment" data-fragment-index="2"> repetition </li> 

</div>


<div class="column" style="float:right; width:80%; text-align: right">

<img class="fragment" data-fragment-index="3" src="./my_figures/spectrogram_cheap.png" width="100%" data-audio-src="./audio/cheap_clean.wav" data-audio-advance=-1> 

</div>

## Harmonicity

<img class="fragment" data-fragment-index="4" src="./my_figures/spectrogram_arctic_a0002.png" width="80%" data-audio-src="./audio/sentences/arctic_a0002.wav" data-audio-advance=-1> 



## Which features conveyed by speech sounds?

- Temporal fine structure (TFS)
- Envelope information (ENV)

<li class ="fragment" data-fragment-index="0" data-audio-src="./audio/speech-davidvocoder.wav" data-audio-advance=-1> Play 6 Channel vocoder </li>
<li class ="fragment" data-fragment-index="1" data-audio-src="./audio/david.wav" data-audio-advance=-1> Play Original </li>

<img src="./my_figures/temporal-example.png" width="70%"> 



## Binaural cues

When listening to sounds, we rely on three mechanisms for both sound localization and auditory scene analysis 

<li class ="fragment" data-fragment-index="0"> Interaural level differences (ILDs) </li>

<li class ="fragment" data-fragment-index="1"> Interaural time differences (ITDs) </li>

<li class ="fragment" data-fragment-index="2"> Interaural coherence </li>


## ITDs and ILDs

<img src="./my_figures/itd_ild_cartoon.png" width="60%">


## ILDs

<img class="fragment" data-fragment-index="0" src="./my_figures/azimuth.png" width="40%">
<img class="fragment" data-fragment-index="1" src="./my_figures/azimuth_ir.png" width="50%"> 
<li class="fragment" data-fragment-index="2"> ILDs only useful for frequencies higher than 1200 Hz </li> 
<li class="fragment" data-fragment-index="3"> ILDs can be as large as 20 dB (e.g. Bronkhorst &
Plomp, 1988) </li> 

<li class="fragment" data-fragment-index="3">
ILDs resulting from the **head shadow** provide an advantage by sheltering the ear turned toward the target source from noise from the other side.
However, listeners with unilateral hearing loss struggle when the target sound comes from the impaired side, specially in the presence of background noise.
</li> 


## Localization using ITDs

<img class="fragment" data-fragment-index="0" src="./my_figures/azimuth_itd.png" width="40%">
<img class="fragment" data-fragment-index="1" src="./my_figures/itd_ir.png" width="50%">

<li class="fragment" data-fragment-index="2"> 
ITDs within the physiological range experienced by human listeners about ±760 μs; (e.g. Constan and Hartmann (2003) and Hartmann and Macaulay (2014)) </li>

<li class="fragment" data-fragment-index="3">
ITDs (using fine structure) are useful for frequencies below  1500 Hz
</li>

<li class="fragment" data-fragment-index="4">
ITDs in the envelope of the signal are also used to determine the location of a source in both lower and higher frequencies.
</li>

## Temporal fine structure and envelope ITDs

<div class="column" style="float:left; width:50%; text-align: center">
<img class="fragment" data-fragment-index="0" src="./my_figures/p1_0.00_p2_-1.57_mi_0.0_mp1_0.00_mp2_0.00_cf_500.0_mf_32.0.png" width="50%" data-audio-src="./audio/p1_0.00_p2_-1.57_mi_0.0_mp1_0.00_mp2_0.00_cf_500.0_mf_32.0.wav" data-audio-advance=-1>
<p class="fragment" data-fragment-index="0"> 500 Hz tone TFS left-to-right </p>

<img class="fragment" data-fragment-index="1" src="./my_figures/p1_0.00_p2_-0.00_mi_1.0_mp1_0.00_mp2_-1.57_cf_500.0_mf_32.0.png" width="50%" data-audio-src="./audio/p1_0.00_p2_-0.00_mi_1.0_mp1_0.00_mp2_-1.57_cf_500.0_mf_32.0.wav" data-audio-advance=-1>
<p class="fragment" data-fragment-index="1"> Modulated 500 Hz tone <br> with envelope itd (ipd) left-to-right</p>
</div>

<div class="column" style="float:right; width:50%; text-align: center">
<img class="fragment" data-fragment-index="2" src="./my_figures/p1_0.00_p2_-1.57_mi_0.0_mp1_0.00_mp2_0.00_cf_4000.0_mf_32.0.png" width="50%" data-audio-src="./audio/p1_0.00_p2_-1.57_mi_0.0_mp1_0.00_mp2_0.00_cf_4000.0_mf_32.0.wav" data-audio-advance=-1>
<p class="fragment" data-fragment-index="2"> 4000 Hz tone TFS itd (ipd) left-to-rigth </p>

<img class="fragment" data-fragment-index="3" src="./my_figures/p1_0.00_p2_0.00_mi_1.0_mp1_0.00_mp2_-1.57_cf_4000.0_mf_32.0.png" width="50%" data-audio-src="./audio/p1_0.00_p2_-0.00_mi_1.0_mp1_0.00_mp2_-1.57_cf_4000.0_mf_32.0.wav" data-audio-advance=-1>
<p class="fragment" data-fragment-index="3"> Modulated 4000 Hz tone <br> with envelope itd left-to-rigth </p>
</div>



## Localization using ITDs

<img src="./my_figures/wave_length.png" width="40%">

<li class="fragment" data-fragment-index="1"> 
ITDs are ambiguous for frequencies which half-wavelengths are smaller than the head 'diameter'.
</li>


## Localization on the vertical plane

<img class="fragment" data-fragment-index="0" src="./my_figures/zenith.png" width="40%">
<img class="fragment" data-fragment-index="1" src="./my_figures/zenith_ir.png" width="50%">

<li class="fragment" data-fragment-index="2"> The main cues are provided by diffraction patterns by the head and pinna, which result in characteristic spectral dips 
</li>

<li class="fragment" data-fragment-index="3"> 
Hearing-impaired subjects have wider cochlear filters (increasing spectral smearing), preventing them to localize a source in the vertical plane.
</li>

## The precedence effect

<img src="./my_figures/precedence_effect_cartoon.png" width="80%">
<br>
(Dietz et al. 2013)

<li class="fragment" data-fragment-index="0"> 
A single auditory event is perceived at the direction of the first direct wave front (2 and 50 ms later, even when reflections are 10 dB louder)
</li>

<li class="fragment" data-fragment-index="1">
Asymmetric hearing loss, hearing impairment and ageing negatively affect the precedence effect (Akeroyd and Guy,
2011). This can only partially restored by hearing aids.
</li>
 
# Binaural processing and speech understanding

## Binaural Redundancy

- Loudness doubles when the two ears are used instead of one ear for a sound coming from the front of the listener (a single ear would require an increase of about 10 dB; Fletcher and Munson, 1933)

- Just noticeable differences in intensity and frequency improve with signal redundancy

- Speech recognition in the presence of background noise improves (Marrone 2008, Neher 2009)

- Hearing impairment may lead to a slightly weaker binaural benefit in patients  (Dillon, 2001)

- Binaural stimulation sounds can be louder than with a monaural presentation without causing discomfort (even true for CI-treated patient)

## Binaural Release from Masking (or Binaural Squelch; or Hirsh effect)

<img class="fragment" data-fragment-index="0" src="./my_figures/bmld_cartoon_n0s0.png" width="45%" data-audio-src="./audio/convolved/sinusoidal_arctic_a0002_d_type_noise_target_0_d1_0_d2_0_snr_-10.wav" data-audio-advance=-1 data-audio-advance=-1 >
<img class="fragment" data-fragment-index="1" src="./my_figures/bmld_cartoon_n0spi.png" width="45%" data-audio-src="./audio/convolved/sinusoidal_arctic_a0002_d_type_noise_target_0_d1_90_d2_-90_snr_-10.wav" data-audio-advance=-1 data-audio-advance=-1>

<li class="fragment" data-fragment-index="2"> Binaural release from masking may improve detection threshold up to about 16 dB for frequencies around 250 Hz and around 3 dB at 1500 Hz </li>

## Spatial Release from Masking  

<img class="fragment" data-fragment-index="0" src="./my_figures/bmld_cartoon_srm_front.png" width="45%" data-audio-src="./audio/convolved/convolved_arctic_a0002_d_type_speech_target_0_d1_0_d2_0_snr_0.wav" data-audio-advance=-1 data-audio-advance=-1 >
<img class="fragment" data-fragment-index="1" src="./my_figures/bmld_cartoon_srm.png" width="45%" data-audio-src="./audio/convolved/convolved_arctic_a0002_d_type_speech_target_0_d1_-80_d2_80_snr_0.wav" data-audio-advance=-1 data-audio-advance=-1>

<li class="fragment" data-fragment-index="2"> 
Binaural release from masking may improve detection thresholds up to 12 dB for multiple speech interferers (Jones and Litovsky, 2011), and facilitates source segregation provided that streaming can build up and natural onset cues are present (Drennan, Gatehouse, and Lever, 2003).
</li>

<li class="fragment" data-fragment-index="3"> 
Segregation is always better for the combination of both ITDs and ILDs cues (Culling, Hawley, and Litovsky 2004)
</li>

<li class="fragment" data-fragment-index="4"> 
A separation of only 10° between two voices is already strong enough to allow segregation  (Brungart and Simpson, 2007).
</li>


<li class="fragment" data-fragment-index="5"> ITD is a critical spatial cue for sound localization and speech perception in noise (Bronkhorst &
Plomp, 1988; Wightman & Kistler, 1992).
</li>


# What and where 


##

Input to one ear

<img src="./my_figures/spectrogram_cheap.png" width="45%">
<p class="fragment" data-fragment-index="1"> Output at the auditory nerve level </p>
<img class="fragment" data-fragment-index="1" src="./my_figures/rater.png" width="45%">

## Auditory pathway

<img src="./my_figures/auitorypath.png" width="50%">


## ILD pathway

<img src="./my_figures/ild_pathway.svg" width="80%">

(Grothe et al. 2010)

<li class="fragment" data-fragment-index="0">
The initial site of ILD processing is generally considered to be the LSO.
</li>

<li class="fragment" data-fragment-index="1">
LSO are innervated by direct excitatory ipsilateral inputs from spherical bushy cells and inhibitory inputs indirectly originating from the globular bushy cells.
</li>

## ITD pathway

<img src="./my_figures/itd_pathway.svg" width="80%">

(Grothe et al. 2010)

<li class="fragment" data-fragment-index="0">
The initial site of ITD processing is considered to be the MSO.
</li>

<li class="fragment" data-fragment-index="1">
MSO are innervated by direct excitatory ipsi- anc contra-lateral inputs from spherical bushy cells and inhibitory inputs indirectly originating from the globular bushy cells.
</li>


# Objective detection of binaural processing in humans

## Electroencephalogram (EEG)

<img src="./my_figures/Luck_2005_erp.png" width="60%">

(From Luck 2005)

##
<img src="./my_figures/eeg_setting.png" width="60%">

<li>
EEG allows to record electrical activity of the brain by means of electrodes placed along the scalp.
</li>

<img src="./my_figures/software_biosemi_01_large.jpg" width="60%">

##

<li class="fragment" data-fragment-index="3">
Objective measures of binaural processing can be obtained by using stimuli where the temporal fine structure is manipulated so that the perceived location of the sound image changes periodically (e.g. 6.7 Hz) over time. 

<img src="./my_figures/example.png" width="40%">

(Undurraga et al. 2016)
</li>

##  Interaural-phase modulation

- The size of the interaural phase difference (relative ITD with respect to the frequency of the sound) will determine the lateralization of the sound.

<br>

||||
|-|-|-|
|<video width="320" height="240" controls source src="./my_figures/example-90.0_90.0.mp4" type="video/mp4">| <video width="320" height="240" controls source src="./my_figures/example-45.0_45.0.mp4" type="video/mp4">|<video width="320" height="240" controls source src="./my_figures/example-22.5_22.5.mp4" type="video/mp4">|

## Interaural-phase modulation following-responses (IPM-FR)

<ul class="fragment" data-fragment-index="0">
<img src="./my_figures/undurraga_2016.svg" width="100%">
</ul>

<li class="fragment" data-fragment-index="1">
By swithcing the "sound image" from left to right at 6.7 Hz, a strong steady-state response is evoked at that particular frequency
</li>

<li class="fragment" data-fragment-index="2">
IPM-FRs are larger when the stimuli causes a strong lateralization percept.
</li>


## How well do objective IPM-FR and behavioural measurements correlate with each other?

Behavioural task

<ul class="fragment" data-fragment-index="0">
<img src="./my_figures/undurraga_2016_behavioural_paradigm_trial_1.png" width="60%">
</ul>
<ul class="fragment" data-fragment-index="1">
<img src="./my_figures/undurraga_2016_behavioural_paradigm_trial_2.png" width="60%">
</ul>
<ul class="fragment" data-fragment-index="2">
<img src="./my_figures/undurraga_2016_behavioural_paradigm_trial_3.png" width="60%">
</ul>

<ul class="fragment" data-fragment-index="3">
The level of the masking noise to perform at chance is determined and compared and then compared with brain responses (IPM-FR)
</ul>

## Behavioural vs. Objective measures


<ul class="fragment" data-fragment-index="3">
<img src="./my_figures/spectral-magnitude-thr-ipd-ave.svg" width="100%">
</ul>

<li class="fragment" data-fragment-index="4">
The correlation between objective and behavioural measurements is excellent (r = 0.96)
</li>


## The effect of age and hearing loss in objective binaural measures

- Does the IPM-FR differ between healthy and impaired hearing listeners?

<ul class="fragment" data-fragment-index="0"> Vercammen et al. (2018) </ul>

<img src="./my_figures/vercommen_pta.png" width="60%" class="fragment" data-fragment-index="0">

<li class="fragment" data-fragment-index="1">
Interaural difference in hearing thresholds near the stimulation frequency (500 Hz) did not exceed 10 dB HL.
</li>

<li class="fragment" data-fragment-index="2">
The stimulus level was 65 dB SPL for normal listeners.
</li>

<li class="fragment" data-fragment-index="3">
The stimulus level for hearing impaired listeners was adjusted using visual analogue scale (“inaudible”, “very soft”, “soft”, “comfortable”, “loud”, “very loud”, and “uncomfortably loud”). 
The subjective loudness was adjusted to match that of normal listeners.
</li>

##

Stimuli were amplitude modulated tones (500 Hz) using IPDs ranging from 0 (diotic) to 180 (dichotic).

<img src="./my_figures/vercommen_paradigm_fig.png" width="40%">

<img src="./my_figures/vercommen_paradigm.png" width="40%" class="fragment" data-fragment-index="0">


##

<img src="./my_figures/vercommen_ipm-fr.png" width="30%">
<img src="./my_figures/vercammena_ipm_fr_data.png" width="50%">

<li class="fragment" data-fragment-index="0">
Binaural processing of low frequency sounds is significantly deteriorated by age and hearing status.
</li>

<li class="fragment" data-fragment-index="1">
A larger IPM-FR dynamic range was associated with lower (better) IPD discrimination.
</li>


# Models for ITD processing

## Straightness weighting


<img src="./my_figures/thompson_jeffress_2006.png" width="80%">

- Peaks of the cross-correlograms at –1.5 ms < peaks closer to 0 - central weighting.

- Correct localization estimated from second processing level (in the inferior colliculus, gray curve on top of each panel).


## The $\pi$-limit 

<!-- - Findings in small mammals suggest that this sort of model may be implausible, even with the inclusion of straightness weighting.  -->
<img src="./my_figures/thompson_2006_pi_limit.png" width="80%">

- ITD detectors in the mammalian brain restricted to $\approx$ half a cycle of best frequency. 
- Frequency-dependent weighting for centrality


##

Thompson et al. (2006) 

<img src="./my_figures/thompson_2006_2.png" width="80%">

- Inferior colliculus consistent with the $\pi$␲-limit.
- Cortical responses to sounds with ITDs within the $\pi$-limit are in line with the predictions of both models. 
- However, neural activation is bilateral for "long" ITDs, despite these being perceived as clearly lateralized
- Long ITDs leads to higher activation in cortex than processing of short ITDs.


## 

Kriegstein et al. (2008)

<img src="./my_figures/kriegstein_2008_2.png" width="40%">

- human and animal Magnetoencephalography (MEG) recordings as well as computational models with ITD detectors restricted either to ±π or with their best ITDs well beyond ±π but with strong weighting to ITDs near zero - *“central-weighting”* agree with a limited range of ITD detectors (Salminen et al., 2018; Stern and Shear, 1996).


## Neural representation of asymmetric interaural time modulations (ITMs)

- ITMs: 0/ + 0.5 ms, 0/ + 1.5 ms, 0/ + 2.0 ms, 0/ + 3.0 ms, 0/ + 4.0 ms.

<img src="./my_figures/average-itds-zero-ref.png" width="40%">
<img src="./my_figures/itm-fr-asym-scalp.png" width="11%">


## 

<img src="./my_figures/itm-fr-coherence-zero-ref.png" width="60%">
<img src="./my_figures/itm-fr-zero-ref-diff.png" width="50%">



## Testing the independence of ITD detectors

- ITMs: −0.5/ + 0.5 ms, −0.5/ + 1.5 ms, and +0.5/ + 1.5 ms.

<div class="column" style="float:left; width:50%; text-align: center">
<img src="./my_figures/pi_s_c_-500us.png" width="30%">
<br>
<img src="./my_figures/pi_s_c_-500us.png" width="30%">
<br>
<img src="./my_figures/pi_s_c_500us.png" width="30%">
</div>

<div class="column" style="float:right; width:50%; text-align: center">
<img src="./my_figures/pi_s_c_500us.png" width="30%">
<br>
<img src="./my_figures/pi_s_c_1500us.png" width="30%">
<br>
<img src="./my_figures/pi_s_c_1500us.png" width="30%">
</div>

<br>
The predictions are:

- the ITM between −0.5 ms and −0.5 ms should be less adapted than the ITM between −0.5 ms and +1.5 ms.
- the ITM between +0.5 ms and +1.5 ms should be less adapted than the ITM between −0.5 ms and +1.5 ms

## Results 

<div class="column" style="float:left; width:50%; text-align: center">
<img src="./my_figures/ipm-fr-adaptation-gfp.png" width="70%">
</div>

<div class="column" style="float:right; width:50%; text-align: center">
<img src="./my_figures/itm-fr-adaptation-gfp-diff.png" width="80%">
</div>


## Neural representation of symmetric ITMs

- ITMs: −0.5/ + 0.5 ms, −1.0/ + 1.0 ms, −1.5/ + 1.5 ms, −2.0/ + 2.0 ms, −2.5/ + 2.5 ms, −3.0/ + 3.0 ms, −4.0/ + 4.0 ms

<div class="column" style="float:left; width:50%; text-align: right">
<img src="./my_figures/average-itds-coherence.png" width="100%">
</div>

<div class="column" style="float:right; width:50%; text-align: center">
<img src="./my_figures/scalp_map_coherence.png" width="16%">
</div>


## Results

<div class="column" style="float:left; width:40%; text-align: center">
<img src="./my_figures/itm-fr-coherence.png" width="100%">

- Significant effect of the ITM (F(7,56.4) = 6.6, p < 0.001).

- The damping pattern may result by the trade-off between conflicting envelope and fine structure ITD. 
Whilst the envelope has a consistent ITD across all conditions, the interaural phase difference (IPD) of the 500 Hz carrier does not.
</div>

<div class="column" style="float:left; width:60%; text-align: left">
<img class="fragment" data-fragment-index="1" src="./my_figures/itd_env_0_5.tiff" width="40%">
<img class="fragment" data-fragment-index="2" src="./my_figures/itd_env_1_0.tiff" width="40%">

<img class="fragment" data-fragment-index="3" src="./my_figures/itd_env_1_5.tiff" width="40%">
<img class="fragment" data-fragment-index="4" src="./my_figures/itd_env_2_0.tiff" width="40%">
</div>


<li class="fragment" data-fragment-index="1">
At −0.5/ + 0.5 ms (−90 ◦ / 90 ◦ IPD), the IPD of the centre frequency is consistent with the envelope ITD.
</li>

<li class="fragment" data-fragment-index="2">
At −1.0/ + 1.0 ms (−180 ◦ / 180 ◦ IPD), the IPD of the centre frequency is ambiguous.
</li>

<li class="fragment" data-fragment-index="3">
At −1.5/ + 1.5 ms (−270 ◦ / 270 ◦ IPD), the IPD of the centre frequency is conflicting with the envelope ITD, i.e. the fine structure IPD leads in the right ear whilst the envelope ITD leads on the left ear.
</li>

<li class="fragment" data-fragment-index="4">
At −2.0/ + 2.0 ms (−360 ◦ / 360 ◦ IPD), there is not IPD in the fine structure at the centre frequency, and so no conflicting cues.
</li>

## 


## Conclusions


<li class="fragment" data-fragment-index="0">
Interaural level and, in particular, interaural time differences conveyed by low frequencies (< 1500 Hz) are critical for speech understanding and segregation in the cocktail party problem (e.g. Swaminathan et al. 2016)
</li>

<!-- <li class="fragment" data-fragment-index="0"> -->
<!-- Interaural time differences are the main cue used to determine the horizontal position below 1.5 kHz and provide an strong cue for segreating multiple sources. -->
<!-- </li> -->

<li class="fragment" data-fragment-index="1">
Hearing-impaired listeners who typically find noisy environments disproportionately difficult for understanding speech, whether they wear a hearing aid or not (Killion 1997; Moore 1998).
</li>

<li class="fragment" data-fragment-index="2">
Similar problems are encountered by users of cochlear implants, again owing to the limited frequency resolution available with implants (Clark 2003).
</li>

<li class="fragment" data-fragment-index="3"> 
Binaural release from masking may improve detection thresholds up to 12 dB for multiple speech interferers (Jones and Litovsky, 2011), and facilitates source segregation provided that streaming can build up and natural onset cues are present (Drennan, Gatehouse, and Lever, 2003).
</li>

<li class="fragment" data-fragment-index="4"> 
Segregation is always better for the combination of both ITDs and ILDs cues (Culling, Hawley, and Litovsky 2004)
</li>

<li class="fragment" data-fragment-index="5">
Objective measures of binaural processing agree with difficulties to localize and understand speech in observed in hearing impaired and elder listeners.
</li>




## Acknowledgments

- Organization for inviting us!!!
- Australian Research Council [project number FL160100108]
- Thanks to you for "listening"

<img src="./my_figures/ahh.jpg" width="60%">
<br>
<img src="./my_figures/ahh_team.jpg" width="60%">


<!-- ## References -->
