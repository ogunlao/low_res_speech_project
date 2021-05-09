from pydub import AudioSegment as am
import multiprocessing
from pathlib import Path
import os
import glob

from utils import make_dirs

def resample_audios(audio_files, src_path, 
                   dest_path, sample_slice,
                   dest_frame_rate, args):
    """Resamples the audio to the dest_frame_rate and also save the duration

    Args:
        audio_files (list): [list of audio files to be resampled]
        src_path (str)
        dest_path (str)
        sample_slice (int)
        dest_frame_rate (int)
        args (dict)
    """
    save_duration_file = args.get('DURATION_SAV_FILE', '')
    save_duration_file = save_duration_file+str(sample_slice)+'.csv'

    download_path = dest_path + os.sep + ".."
    with open(os.path.join(download_path, save_duration_file), "w+") as f:                    
        for audio_file in audio_files:
            wav_file = str(Path(audio_file).stem) + ".wav"
            dest_path = os.path.join(dest_path, wav_file)
            
            # convert mp3 to wav
            try:                                                          
                sound = am.from_mp3(audio_file)
                
                # calculate duration of wav file
                duration = sound.duration_seconds
                
                sound = sound.set_frame_rate(dest_frame_rate)      
                sound.export(dest_path, format="wav")

                # save duration in file
                f.write(f"{audio_file},{duration}")
                f.write("\n")
                
            except Exception:
                print(f'File {audio_file} unable to be processed')
            
            
def sample_audio(args):
    """Download audio files, and resamples to required sampling rate

    Args:
        args (dict): Contains the arguments required for parsing directories
    """
    curr_path = Path(__file__).parent.absolute()
    download_path = str(curr_path)+os.sep+'..'+os.sep+args.get('DATA_FOLDER')

    raw_audio_paths = args.get('RAW_AUDIO_PATH')
    audio_files = glob.glob(download_path + os.sep + str(raw_audio_paths) + os.sep + "*.mp3")
    
    sample_source_path = download_path + os.sep + str(raw_audio_paths)
    sample_dest_path = download_path+os.sep+args.get('SAMPLED_DATA_FOLDER')
    make_dirs(sample_dest_path)
    
    sampling_rate = args.get("SAMPLING_RATE", 16000)

    if len(os.listdir(sample_dest_path)) == 0: # Check if empty..
        processes = []
        
        num_cpu = multiprocessing.cpu_count()
        total_splits = len(audio_files)//num_cpu
        
        print("{num_cpu} cpu(s) available for parallel resampling of files")
        for i in range(num_cpu-1):
            current_split = audio_files[total_splits*i: total_splits*i+total_splits]
            proc = multiprocessing.Process(target=resample_audios, args=(current_split,
                                                                    sample_source_path,
                                                                    sample_dest_path, i, 
                                                                    sampling_rate, args,))
            processes.append(proc)
            proc.start()
        final_split = audio_files[total_splits*(num_cpu-2):]
        proc = multiprocessing.Process(target=resample_audios, args=(current_split,
                                                                sample_source_path,
                                                                sample_dest_path, num_cpu-1, 
                                                                sampling_rate, args,))
        processes.append(proc)
        proc.start()

        # complete the processes
        for proc in processes:
            proc.join()
            
        print('Sampling done')