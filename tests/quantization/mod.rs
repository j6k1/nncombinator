use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use rand::prelude;
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use nncombinator::activation::{ReLuBuilder, SoftMaxBuilder};
use nncombinator::arr::Arr;
use nncombinator::device::DeviceCpu;
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::{AddLayer, BatchForward, BatchTrain, ForwardAll};
use nncombinator::layer::input::{QuantizedInputLayer};
use nncombinator::layer::linear::{QuantizedLinearLayerBuilder};
use nncombinator::layer::logging::LoggingLayer;
use nncombinator::layer::scale::ScalingLayerBuilder;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::layer::quantization::DequantizeLayerBuilder;
use nncombinator::lossfunction::CrossEntropyMulticlass;
use nncombinator::optimizer::AdamWBuilder;
use crate::common::{assert_backward_all, assert_batch_backward, assert_batch_forward, assert_batch_loss, assert_batch_pre_train, assert_forward_all, assert_loss, assert_pre_train, assert_update_weight};

#[test]
fn test_mnist_for_quntization_cpu() {
    let mut rnd = prelude::thread_rng();

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/512f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(256f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:QuantizedInputLayer<i16,Arr<i16,{ 28*28 }>,_,_,255,{ 28*28 }> = QuantizedInputLayer::new(&device).unwrap();

    let optimizer_builder = AdamWBuilder::new(&device).lr(0.001).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);

        QuantizedLinearLayerBuilder::<i8,i16,{ 28*28 },512>::new().build(l,&device,
                                                         || n1.sample(&mut rnd), || 0.,
                                                         &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);

        ActivationLayer::new(l,ReLuBuilder::new(&device),&device)
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_loss(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);
        assert_batch_loss(&l);

        QuantizedLinearLayerBuilder::<i8,i16,512,256>::new().build(l,&device,
                                                   || n2.sample(&mut rnd), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);

        ActivationLayer::new(l,ReLuBuilder::new(&device),&device)
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_loss(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);
        assert_batch_loss(&l);

        QuantizedLinearLayerBuilder::<i8,i16,256, 10>::new().build(l, &device,
                                                            || n3.sample(&mut rnd), || 0.,
                                                            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);

        DequantizeLayerBuilder::<f32,Arr<f32,10>>::new().build(l, &device).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);

        ActivationLayer::new(l,SoftMaxBuilder::new(&device),&device)
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_loss(&l);
        assert_batch_forward(&l);
        assert_batch_backward(&l);
        assert_batch_pre_train(&l);
        assert_batch_loss(&l);

        LinearOutputLayer::new(l,&device).unwrap()
    });

    assert_forward_all(&net);
    assert_pre_train(&net);
    assert_backward_all(&net);
    assert_update_weight(&net);
    assert_batch_forward(&net);
    assert_batch_backward(&net);
    assert_batch_pre_train(&net);

    let mut teachers:Vec<(usize,PathBuf)> = Vec::new();

    for n in 0..10 {
        for entry in fs::read_dir(Path::new("mnist")
            .join("mnist_png")
            .join("training")
            .join(n.to_string())).unwrap() {
            let path = entry.unwrap().path();

            teachers.push((n,path));
        }
    }

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().take(60000).collect::<Vec<(usize,PathBuf)>>();

    let start_time = Instant::now();

    for _ in 0..10 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(64) {
            let batch_data = teachers.iter().map(|(n, path)| {
                count += 1;

                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();

                let n = *n;

                let mut input = Arr::<i16, 784>::new();

                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as i16;
                }

                let mut expected = Arr::new();

                expected[n as usize] = 1.0;

                (expected, input)
            }).fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<i16, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
    }

    let elapsed = start_time.elapsed();

    let elapsed = elapsed.as_secs() as u64 * 1000 + elapsed.subsec_millis() as u64;

    println!("processing time is {} secs.",elapsed as f64 / 1000.);

    let mut tests: Vec<(usize, PathBuf)> = Vec::new();

    for n in 0..10 {
        for entry in fs::read_dir(Path::new("mnist")
            .join("mnist_png")
            .join("testing")
            .join(n.to_string())).unwrap() {
            let path = entry.unwrap().path();

            tests.push((n, path));
        }
    }

    tests.shuffle(&mut rng);

    let count = tests.len();

    for (n, path) in tests.iter() {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<i16, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as i16;
        }

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        if n == r {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 80.)
}
