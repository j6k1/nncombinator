use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::DerefMut;
use std::path::Path;
use std::rc::Rc;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Sigmoid};
use nncombinator::arr::Arr;
use nncombinator::device::DeviceGpu;
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::{AddLayer, BatchTrain, Step};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::linear::LinearLayerBuilder;
use nncombinator::layer::logging::LoggingLayer;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::lossfunction::CrossEntropy;
use nncombinator::optimizer::MomentumSGDBuilder;
use nncombinator::scheduler::{LambdaLR, LinearWarmupLR, Scheduler};
use crate::common::{assert_backward_all, assert_batch_backward, assert_batch_forward, assert_batch_loss, assert_batch_pre_train, assert_forward_all, assert_loss, assert_on_step, assert_pre_train, assert_step, assert_update_weight, SHARED_MEMORY_POOL};

#[test]
fn test_scheduler_seq() {
    let mut scheduler = LinearWarmupLR::new(10, 0.01, 0.0)
                                        .seq(10, LinearWarmupLR::new(20, 0.01, 0.0))
                                        .seq(20,LambdaLR::new(0.01, |_| Ok(0.001)));

    assert_eq!(scheduler.schedule_frequently(0.01, 0,0).unwrap(), 0.0);
    assert_eq!(scheduler.schedule_frequently(0.01, 5,9).unwrap(), 0.01 * (9.0 / 10.0));
    assert_eq!(scheduler.schedule_frequently(0.01, 10,2).unwrap(), 0.01 * (2.0 / 20.0));
    assert_eq!(scheduler.schedule_frequently(0.01, 11,12).unwrap(), 0.01 * (12.0 / 20.0));
    assert_eq!(scheduler.schedule_frequently(0.01, 29,18).unwrap(), 0.01 * (18.0 / 20.0));
    assert_eq!(scheduler.schedule(0.01, 30).unwrap(), 0.00001);
    assert_eq!(scheduler.schedule(0.01, 35).unwrap(), 0.00001);
    assert_eq!(scheduler.schedule(0.01, 40).unwrap(), 0.00001);
    assert_eq!(scheduler.schedule(0.01, 50).unwrap(), 0.00001);

    assert_eq!(scheduler.schedule(0.01, 0).unwrap(), 0.01);
    assert_eq!(scheduler.schedule(0.02, 5).unwrap(), 0.02);
    assert_eq!(scheduler.schedule(0.03, 10).unwrap(), 0.03);
    assert_eq!(scheduler.schedule(0.04, 11).unwrap(), 0.04);
    assert_eq!(scheduler.schedule(0.05, 29).unwrap(), 0.05);
}
#[test]
fn test_scheduler() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,14>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.001);

    let mut net = net.add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        let rnd = rnd.clone();
        LinearLayerBuilder::<14,100>::new().build(l,&device,
                                                  move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        let rnd = rnd.clone();
        LinearLayerBuilder::<100,100>::new().build(l,&device,
                                                   move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        let rnd = rnd.clone();
        LinearLayerBuilder::<100, 1>::new().build(l, &device,
                                                  move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        let l = LoggingLayer::new(l,&device);
        l
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_on_step(&l);

        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer(|l| {
        assert_forward_all(&l);
        assert_pre_train(&l);
        assert_backward_all(&l);
        assert_loss(&l);
        assert_update_weight(&l);
        assert_batch_forward(&l);
        assert_batch_pre_train(&l);
        assert_batch_backward(&l);
        assert_batch_loss(&l);
        assert_on_step(&l);

        LinearOutputLayer::new(l,&device).unwrap()
    });

    assert_forward_all(&net);
    assert_pre_train(&net);
    assert_backward_all(&net);
    assert_update_weight(&net);
    assert_batch_forward(&net);
    assert_batch_pre_train(&net);
    assert_batch_backward(&net);
    assert_step(&net);

    let mut teachers:Vec<(bool,Vec<f32>)> = Vec::new();

    let mut reader =  BufReader::new(
        File::open(Path::new("data")
            .join("weather")
            .join("training")
            .join("weather.csv")).unwrap());

    let mut line:String = String::new();

    loop {
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }

        let columns = line.trim().split(',').map(|c| c.to_string()).collect::<Vec<String>>();

        line.clear();

        if columns.len() != 16 {
            continue;
        }

        let t = columns[1].find("晴").is_some();

        let columns = columns.iter().skip(2)
            .filter(|c| !c.parse::<f32>().is_err())
            .map(|c| c.parse::<f32>().unwrap() / 10000.)
            .collect::<Vec<f32>>();
        if columns.len() < 14 {
            continue;
        }

        teachers.push((t,columns));
    }

    let mut rng = rand::thread_rng();

    let lossf = CrossEntropy::new();

    let mut iter = teachers.chunks_mut(10).take(10);

    while let Some(teachers) = iter.next()  {
        teachers.shuffle(&mut rng);

        let mut train_data = Vec::new();

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f32, 14>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[0] = if t {
                1.
            } else {
                0.
            };

            train_data.push((expected,input));
        }

        let train_data = train_data.into_iter().fold((Vec::new(),Vec::new()),|mut acc,(e,input)| {
            acc.0.push(e);
            acc.1.push(input);

            acc
        });
        let _ = net.batch_train(train_data.0.into(),train_data.1.into(),&lossf).unwrap();
        net.step().unwrap();
    }

    assert!(true);
}
