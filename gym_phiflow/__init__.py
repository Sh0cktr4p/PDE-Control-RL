from gym.envs.registration import register

register(
    id='burger-v00',
    entry_point='gym_phiflow.envs:BurgerEnvTwo',
)

register(
    id='burger-v01',
    entry_point='gym_phiflow.envs:BurgerEnvThree',
)

register(
    id='burger-v02',
    entry_point='gym_phiflow.envs:BurgerEnvContComplete',
)

register(
    id='burger-v03',
    entry_point='gym_phiflow.envs:BurgerEnvTwoRel',
)

register(
    id='burger-v04',
    entry_point='gym_phiflow.envs:BurgerEnvThreeRandom'
)

register(
    id='burger-v05',
    entry_point='gym_phiflow.envs:BurgerEnvThreeReachable'
)

register(
    id='burger-v06',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteRandom'
)

register(
    id='burger-v07',
    entry_point='gym_phiflow.envs:BurgerEnvThreeThreeReachable'
)

register(
    id='burger-v08',
    entry_point='gym_phiflow.envs:BurgerEnvContEightReachable'
)

register(
    id='burger-v09',
    entry_point='gym_phiflow.envs:BurgerEnvThreeThreeReachableTime'
)

register(
    id='burger-v10',
    entry_point='gym_phiflow.envs:BurgerEnvContEightReachableTime'
)

register(
    id='burger-v11',
    entry_point='gym_phiflow.envs:BurgerEnvContSixteen2DReachable'
)

register(
    id='burger-v15',
    entry_point='gym_phiflow.envs:BurgerEnvTwoReachableSync'
)

register(
    id='burger-v100',
    entry_point='gym_phiflow.envs:BurgerEnvThreeTwoReachable'
)

register(
    id='burger-v101',
    entry_point='gym_phiflow.envs:BurgerEnvContTwoReachable'
)

register(
    id='burger-v102',
    entry_point='gym_phiflow.envs:BurgerEnvThreeFourReachable'
)

register(
    id='burger-v103',
    entry_point='gym_phiflow.envs:BurgerEnvContFourReachable'
)

register(
    id='burger-v104',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstant'
)

register(
    id='burger-v105',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstantSmoothed'
)

register(
    id='burger-v106',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstantPow'
)

register(
    id='burger-v107',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstantRel'
)

register(
    id='burger-v108',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstantFin'
)

register(
    id='burger-v109',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstantFinL1'
)

register(
    id='burger-v110',
    entry_point='gym_phiflow.envs:BurgerEnvContCompleteConstantPowL1'
)

register(
    id='burger-v200',
    entry_point='gym_phiflow.envs:BurgerEnvTwoReachableFin'
)

register(
    id='burger-v201',
    entry_point='gym_phiflow.envs:BurgerEnvThreeReachableFin'
)

register(
    id='burger-v202',
    entry_point='gym_phiflow.envs:BurgerEnvContReachableFin'
)

register(
    id='burger-v203',
    entry_point='gym_phiflow.envs:BurgerEnvFourThreeReachableFin'
)

register(
    id='burger-v204',
    entry_point='gym_phiflow.envs:BurgerEnvFourContReachableFin'
)

register(
    id='navier-v00',
    entry_point='gym_phiflow.envs:NavierEnvTwo'
)

register(
    id='navier-v12',
    entry_point='gym_phiflow.envs:NavierEnvContTwenty2DReachable'
)

register(
    id='navier-v14',
    entry_point='gym_phiflow.envs:NavierEnvContComplete2DShapes'
)

register(
    id='navier-v16',
    entry_point='gym_phiflow.envs:NavierEnvContComplete2DShapesObs'
)