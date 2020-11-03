import phi.tf.flow as phiflow
import tensorflow as tf


state = phiflow.Fluid(domain=phiflow.Domain((1024, 1024), boundaries=phiflow.CLOSED), density=phiflow.Noise())
physics = phiflow.IncompressibleFlow(pressure_solver=phiflow.CUDASolver())
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))

input_ph = state.copied_with(density=phiflow.placeholder, velocity=phiflow.placeholder)
output_ph = physics.step(input_ph, 0.5)

feed_dict = lambda s: {k:v for (k, v) in zip(phiflow.struct.flatten(input_ph), phiflow.struct.flatten(s))}

step = lambda s: phiflow.struct.unflatten(sess.run(phiflow.struct.flatten(output_ph), feed_dict=feed_dict(s)), s)

print(state)
print(step(state))

#sess.run(tf.global_variables_initializer())
#sess.graph.finalize()

for _ in range(200000):
    state = step(state)
    print('Step')