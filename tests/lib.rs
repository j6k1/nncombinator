extern crate nncombinator;

extern crate rand;
extern crate rand_xorshift;
extern crate statrs;

use rand::{prelude, Rng, SeedableRng};
use rand::prelude::Distribution;
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, SoftMax};
use nncombinator::arr::Arr;
use nncombinator::device::DeviceCpu;
use nncombinator::layer::{ActivationLayer, AddLayer, AddLayerTrain, InputLayer, LinearLayer, LinearOutputLayer};

#[test]
fn test_mnist() {
    let mut rnd = prelude::thread_rng();
    let mut rnd1 = XorShiftRng::from_seed(rnd.gen());
    let mut rnd2 = XorShiftRng::from_seed(rnd.gen());

    let n1 = Normal::<f32>::new(0.0, 1f32/(2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/(28f32*28f32).sqrt()).unwrap();

    let device = DeviceCpu::new();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>> = InputLayer::new();
    let net = net.add_layer(|l| {
        LinearLayer::<_,_,_,_,{ 28*28 },64>::new(l,&device, move || n1.sample(&mut rnd1), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        LinearLayer::<_,_,_,_,64,10>::new(l,&device, move || n2.sample(&mut rnd2), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });
}
