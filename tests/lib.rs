extern crate nncombinator;

extern crate rand;
extern crate rand_xorshift;
extern crate statrs;

use std::cell::RefCell;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Sigmoid, SoftMax};
use nncombinator::arr::{Arr, DiffArr};
use nncombinator::device::DeviceCpu;
use nncombinator::layer::{ActivationLayer, AddLayer, AddLayerTrain, AscDiffInput, DiffInput, DiffLinearLayer, ForwardAll, ForwardDiff, InputLayer, LinearLayer, LinearOutputLayer, Train};
use nncombinator::lossfunction::{CrossEntropyMulticlass, Mse};
use nncombinator::optimizer::{MomentumSGD};

#[test]
fn test_mnist() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, 1f32/(2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/(28f32*28f32).sqrt()).unwrap();

    let device = DeviceCpu::new();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,_,{ 28*28 },64>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,_,64,10>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

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
    let mut optimizer = MomentumSGD::with_params(0.001,0.9,0.0);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut teachers = teachers.into_iter().take(1000).collect::<Vec<(usize,PathBuf)>>();

    let mut correct_answers = 0;

    for _ in 0..10 {
        teachers.shuffle(&mut rng);

        for (n, path) in teachers.iter() {
            let b = BufReader::new(File::open(path).unwrap()).bytes();

            let pixels = b.map(|b| b.unwrap() as f32 / 255.).take(784).collect::<Vec<f32>>();

            let n = *n;

            let mut input = Arr::<f32, 784>::new();

            for (it, p) in input.iter_mut().zip(pixels.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[n as usize] = 1.0;

            let lossf = CrossEntropyMulticlass::new();

            net.train(expected, input, &mut optimizer, &lossf);
        }
    }

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

    for (n, path) in tests.iter().take(100) {
        let b = BufReader::new(File::open(path).unwrap()).bytes();

        let pixels = b.map(|b| b.unwrap() as f32 / 255.).take(784).collect::<Vec<f32>>();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, p) in input.iter_mut().zip(pixels.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input);

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

        println!("n = {}, answer = {}", n, r);
    }

    println!("correct_answers = {}",correct_answers);

    debug_assert!(correct_answers > 10)
}

#[test]
fn test_weather() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceCpu::new();

    let net:InputLayer<f32,Arr<f32,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
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

    let mut optimizer = MomentumSGD::with_params(0.0001,0.9,0.0);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    teachers.shuffle(&mut rng);

    for _ in 0..1 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f32,14>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[0] = if t {
                1.
            } else {
                0.
            };

            let lossf = Mse::new();

            net.train(expected, input, &mut optimizer, &lossf);
        }
    }

    let mut tests:Vec<(bool,Vec<f32>)> = Vec::new();

    let mut reader =  BufReader::new(
        File::open(Path::new("data")
            .join("weather")
            .join("testing")
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

        tests.push((t,columns));
    }

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f32, 14>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            correct_answers += 1;
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}

#[test]
fn test_weather_by_forward_diff() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceCpu::new();

    let net:InputLayer<f32,DiffInput<DiffArr<f32,14>,f32,14,100>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        DiffLinearLayer::<_,_,_,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
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

    let mut optimizer = MomentumSGD::with_params(0.0001,0.9,0.0);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    teachers.shuffle(&mut rng);

    for _ in 0..1 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f32,14>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[0] = if t {
                1.
            } else {
                0.
            };

            let lossf = Mse::new();

            net.train(expected, DiffInput::NotDiff(input), &mut optimizer, &lossf);
        }
    }

    let mut tests:Vec<(bool,Vec<f32>)> = Vec::new();

    let mut reader =  BufReader::new(
        File::open(Path::new("data")
            .join("weather")
            .join("testing")
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

        tests.push((t,columns));
    }

    let mut s = None;
    let mut prev = Arr::new();

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f32, 14>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        s = if let Some(s) = s {
            let d = input.iter().enumerate().zip(prev.iter())
                                .filter(|((_,&input),&p)| input != p)
                                .map(|((index,&input),&p)| (index,input - p))
                                .fold(DiffArr::new(),| mut acc,(i,d) | {
                acc.push(i,d);
                acc
            });

            prev = input.clone();

            let o = net.asc_diff_input(s);

            Some(net.forward_diff(DiffInput::Diff(d,o)))
        } else {
            prev = input.clone();

            Some(net.forward_diff(DiffInput::NotDiff(input)))
        };

        let r = s.as_ref().map(|r| r.1[0]).unwrap();

        if (t && r >= 0.5) || !t && r < 0.5 {
            correct_answers += 1;
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}
