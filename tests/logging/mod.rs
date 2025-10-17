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
use nncombinator::device::{DeviceGpu};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::{AddLayer, BatchTrain, Train};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::linear::LinearLayerBuilder;
use nncombinator::layer::logging::{LoggingLayerBuilder};
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::lossfunction::CrossEntropy;
use nncombinator::optimizer::MomentumSGDBuilder;
use crate::common::SHARED_MEMORY_POOL;

#[test]
fn test_logger() {
    let (sender,receiver) = std::sync::mpsc::channel();

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
        let rnd = rnd.clone();
        LinearLayerBuilder::<14,100>::new().build(l,&device,
                                                  move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,100>::new().build(l,&device,
                                                   move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100, 1>::new().build(l, &device,
                                                  move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        let mut l = LoggingLayerBuilder::new().build(l,&device).unwrap();

        {
            let sender = sender.clone();

            l.add_forward_logger(move |_| {
                println!("forward");
                sender.send("forward").unwrap();

                Ok(())
            });
        }

        {
            let sender = sender.clone();

            l.add_backward_logger(move |_| {
                println!("backward");
                sender.send("backward").unwrap();

                Ok(())
            });
        }

        {
            let sender = sender.clone();

            l.add_gradient_logger(move |_| {
                println!("gradient");
                sender.send("gradient").unwrap();

                Ok(())
            });
        }

        {
            let sender = sender.clone();

            l.add_batch_forward_logger(move |_| {
                println!("batch forward");
                sender.send("batch forward").unwrap();

                Ok(())
            });
        }

        {
            let sender = sender.clone();

            l.add_batch_backward_logger(move |_| {
                println!("batch backward");
                sender.send("batch backward").unwrap();

                Ok(())
            });
        }

        l
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device)
    });

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

    if let Some(chunk) = iter.next() {
        for (t, columns) in chunk.iter_mut() {
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

            let _ = net.train(expected,input,&lossf).unwrap();
        }
    }

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
    }

    for _ in 0..10 {
        assert_eq!("forward", receiver.recv().unwrap());
        assert_eq!("backward", receiver.recv().unwrap());
        assert_eq!("gradient", receiver.recv().unwrap());
    }

    for _ in 0..9 {
        assert_eq!("batch forward", receiver.recv().unwrap());
        assert_eq!("batch backward", receiver.recv().unwrap());
        assert_eq!("gradient", receiver.recv().unwrap());
    }
}
