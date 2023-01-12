from os import listdir
import json

def ground_truth_to_pose(ground_truth_folder , posefile_name = "pose.txt", transform = None):
    
    files = listdir(ground_truth_folder)
    posefiles = [(ground_truth_folder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
    # posefiles = [(ground_truth_folder +'/'+ ff) for ff in files if (ff.endswith('.json'))]
    posefiles.sort()
    posefolder = ground_truth_folder

    print('Find {} pose files in {}'.format(len(posefiles), ground_truth_folder))
    print(posefiles[0])
    print(int(filter(str.isdigit, posefiles[0])))

    sub1 = "step"
    sub2 = ".camera.png"

    for posefile in posefiles:
        idx1 = posefile.index(sub1)
        idx2 = posefile.index(sub2)

        
        # f = open(posefile)
        # data = json.load(f)
        # print(posefile)
        # print("rotation: ", data['captures'][0]['rotation'])
        # print("position: ", data['captures'][0]['position'])


    # if posefile is not None and posefile!="":
    #     poselist = np.loadtxt(posefile).astype(np.float32)
    #     assert(poselist.shape[1]==7) # position + quaternion
    #     poses = pos_quats2SEs(poselist)
    #     matrix = pose2motion(poses)
    #     motions     = SEs2ses(matrix).astype(np.float32)
    #     # motions = motions / pose_std
    #     assert(len(motions) == len(rgbfiles)) - 1
    # else:
    #     motions = None

    N = len(posefiles) - 1

ground_truth_to_pose("/root/tartan_vo/solo/sequence.0")