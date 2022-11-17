extern crate nncombinator;

extern crate rand;
extern crate rand_xorshift;
extern crate statrs;
extern crate csv;
extern crate lazy_static;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use csv::Reader;
use lazy_static::lazy_static;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Sigmoid, SoftMax, Swish, Tanh};
use nncombinator::arr::{Arr, DiffArr};
use nncombinator::cuda::mem::{Alloctype, MemoryPool};
use nncombinator::device::{DeviceCpu, DeviceGpu};
use nncombinator::error::{TrainingError, UnsupportedOperationError};
use nncombinator::layer::{ActivationLayer, AddLayer, AddLayerTrain, AskDiffInput, BatchForward, BatchTrain, DiffInput, DiffLinearLayer, ForwardAll, ForwardDiff, InputLayer, LinearLayer, LinearOutputLayer, Train};
use nncombinator::lossfunction::{CrossEntropy, CrossEntropyMulticlass, Mse};
use nncombinator::optimizer::{MomentumSGD,SGD};

lazy_static! {
    static ref SHARED_MEMORY_POOL:Arc<Mutex<MemoryPool>> = Arc::new(Mutex::new(MemoryPool::new(Alloctype::Device).unwrap()));
}

#[test]
fn test_mnist() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,10>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
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
    let mut optimizer = SGD::new(0.01);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().take(60000).collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();

                let n = *n;

                let mut input = Arr::<f32, 784>::new();

                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }

                let mut expected = Arr::new();

                expected[n as usize] = 1.0;

                (expected, input)
            }).fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
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
#[test]
fn test_mnist_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,10>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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
    let mut optimizer = SGD::new(0.01);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();

                let n = *n;

                let mut input = Arr::<f32, 784>::new();

                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }

                let mut expected = Arr::new();

                expected[n as usize] = 1.0;

                (expected, input)
            }).fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
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
#[test]
fn test_mnist_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/(28f64*28f64)).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, (2f64/100f64).sqrt()).unwrap();
    let n3 = Normal::<f64>::new(0.0, 1f64/(100f64).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f64,Arr<f64,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,100,10>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();

                let n = *n;

                let mut input = Arr::<f64, 784>::new();

                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f64 / 255.;
                }

                let mut expected = Arr::new();

                expected[n as usize] = 1.0;

                (expected, input)
            }).fold((Vec::<Arr<f64, 10>>::new(), Vec::<Arr<f64, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f64 / count as f64);
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

    let count = tests.len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f64, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f64 / 255.;
        }

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, -1.), |acc, (n, &t)| {
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
#[test]
fn test_weather() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
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

    let mut optimizer = MomentumSGD::new(0.001);

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

            let lossf = CrossEntropy::new();

            net.train(expected, input, &mut optimizer, &lossf).unwrap();
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

        let r = net.forward_all(input).unwrap();

        println!("晴れの確率 {}%",r[0]);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            println!("正解!");
            correct_answers += 1;
        } else {
            println!("不正解...");
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

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,DiffInput<DiffArr<f32,14>,f32,14,100>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        DiffLinearLayer::<_,_,_,DeviceCpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
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

    let mut optimizer = MomentumSGD::new(0.001);

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

            let lossf = CrossEntropy::new();

            net.train(expected, DiffInput::NotDiff(input), &mut optimizer, &lossf).unwrap();
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
                acc.push(i,d).unwrap();
                acc
            });

            prev = input.clone();

            let o = net.ask_diff_input(&s);

            Some(net.forward_diff(DiffInput::Diff(d,o)).unwrap())
        } else {
            prev = input.clone();

            Some(net.forward_diff(DiffInput::NotDiff(input)).unwrap())
        };

        let r = s.as_ref().map(|r| r.1[0]).unwrap();

        if (t && r >= 0.5) || !t && r < 0.5 {
            correct_answers += 1;
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}
#[test]
fn test_diff_learn_error() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,DiffInput<DiffArr<f32,14>,f32,14,100>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        DiffLinearLayer::<_,_,_,DeviceCpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let input = Arr::new();

    let s = net.forward_diff(DiffInput::NotDiff(input)).unwrap();

    let mut optimizer = MomentumSGD::new(0.001);
    let lossf = CrossEntropy::new();

    let mut expected = Arr::new();

    expected[0] = 1.;

    let o = net.ask_diff_input(&s);
    let d = DiffArr::new();

    match net.train(expected, DiffInput::Diff(d,o), &mut optimizer, &lossf) {
        Err(TrainingError::UnsupportedOperationError(e)) => {
            assert_eq!(e,UnsupportedOperationError(
                String::from("Training from difference information is not supported.")
            ));
        },
        _ => assert!(false)
    }
}
#[test]
fn test_diff_learn_error_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,DiffInput<DiffArr<f32,14>,f32,14,100>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        DiffLinearLayer::<_,_,_,DeviceGpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let input = Arr::new();

    let s = net.forward_diff(DiffInput::NotDiff(input)).unwrap();

    let mut optimizer = MomentumSGD::new(0.001);
    let lossf = CrossEntropy::new();

    let mut expected = Arr::new();

    expected[0] = 1.;

    let o = net.ask_diff_input(&s);
    let d = DiffArr::new();

    match net.train(expected, DiffInput::Diff(d,o), &mut optimizer, &lossf) {
        Err(TrainingError::UnsupportedOperationError(e)) => {
            assert_eq!(e,UnsupportedOperationError(
                String::from("Training from difference information is not supported.")
            ));
        },
        _ => assert!(false)
    }
}
#[test]
fn test_weather_batch_train() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
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

    let mut optimizer = MomentumSGD::new(0.01);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    let lossf = CrossEntropy::new();

    for teachers in teachers.chunks_mut(100).take(1000) {
        for _ in 0..5 {
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
            let loss = net.batch_train(train_data.0.into(),train_data.1.into(),&mut optimizer,&lossf).unwrap();

            println!("total_loss = {}",loss);
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

        let r = net.forward_all(input).unwrap();

        println!("晴れの確率 {}%",r[0]);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            println!("正解!");
            correct_answers += 1;
        } else {
            println!("不正解...")
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}
#[test]
fn test_weather_batch_train_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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

    let mut optimizer = MomentumSGD::new(0.01);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    let lossf = CrossEntropy::new();

    for teachers in teachers.chunks_mut(100).take(1000) {
        for _ in 0..5 {
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
            let loss = net.batch_train(train_data.0.into(),train_data.1.into(),&mut optimizer,&lossf).unwrap();

            println!("total_loss = {}",loss);
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

        let r = net.forward_all(input).unwrap();

        println!("晴れの確率 {}%",r[0]);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            println!("正解!");
            correct_answers += 1;
        } else {
            println!("不正解...")
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}
#[test]
fn test_weather_batch_train_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/14f64).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, (2f64/100f64).sqrt()).unwrap();
    let n3 = Normal::<f64>::new(0.0, 1f64/100f64.sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f64,Arr<f64,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut teachers:Vec<(bool,Vec<f64>)> = Vec::new();

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
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap() / 10000.)
            .collect::<Vec<f64>>();
        if columns.len() < 14 {
            continue;
        }

        teachers.push((t,columns));
    }

    let mut optimizer = MomentumSGD::new(0.01);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    let lossf = CrossEntropy::new();

    for teachers in teachers.chunks_mut(100).take(1000) {
        for _ in 0..5 {
            teachers.shuffle(&mut rng);

            let mut train_data = Vec::new();

            for (t, columns) in teachers.iter() {
                let t = *t;

                let mut input = Arr::<f64, 14>::new();

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
            let loss = net.batch_train(train_data.0.into(),train_data.1.into(),&mut optimizer,&lossf).unwrap();

            println!("total_loss = {}",loss);
        }
    }

    let mut tests:Vec<(bool,Vec<f64>)> = Vec::new();

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
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap() / 10000.)
            .collect::<Vec<f64>>();
        if columns.len() < 14 {
            continue;
        }

        tests.push((t,columns));
    }

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f64, 14>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input).unwrap();

        println!("晴れの確率 {}%",r[0]);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            println!("正解!");
            correct_answers += 1;
        } else {
            println!("不正解...")
        }
    }

    println!("rate = {}",correct_answers as f64 / tests.len() as f64 * 100.);
    debug_assert!(correct_answers as f64 / tests.len() as f64 * 100. >= 73.);
}
#[test]
fn test_penguins() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/6f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/50f32.sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,6>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,6,50>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,50,3>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut targets = HashSet::new();

    targets.insert("Culmen Length (mm)");
    targets.insert("Culmen Depth (mm)");
    targets.insert("Flipper Length (mm)");
    targets.insert("Body Mass (g)");
    targets.insert("Delta 15 N (o/oo)");
    targets.insert("Delta 13 C (o/oo)");

    let mut species:HashMap<&str,usize> = HashMap::new();

    species.insert("Adelie",0);
    species.insert("Chinstrap",1);
    species.insert("Gentoo",2);

    let mut teachers:Vec<(usize,Vec<f32>)> = Vec::new();

    let mut reader =  Reader::from_path(
        Path::new("data")
                .join("penguins")
                .join("training")
                .join("penguins_lter.csv")).unwrap();

    let headers = reader.headers().unwrap().iter().map(|c| c.trim().to_string()).collect::<Vec<String>>();

    for row in reader.records() {
        let columns= headers.iter().zip(row.unwrap().iter()).map(|(f,c)| (f.to_string(),c.to_string())).collect::<Vec<(String,String)>>();
        if columns.len() != 17 {
            continue;
        }

        let s = columns[2].1.split(' ').map(|s| s.to_string()).collect::<Vec<String>>();

        let t = species.get(&s[0] as &str).unwrap();

        let columns = columns.iter()
            .filter(|(f,_)| targets.contains(f.as_str().trim()))
            .map(|(_,c)| c.trim().to_owned())
            .filter(|c| !c.parse::<f32>().is_err())
            .map(|c| c.parse::<f32>().unwrap())
            .collect::<Vec<f32>>();
        if columns.len() < 6 {
            continue;
        }
        teachers.push((*t,columns));
    }

    let count = teachers.len() as f32;
    let mut sum = 0f32;

    for (_,row) in teachers.iter() {
        sum += row.iter().fold(0.,|acc,x| acc + x);
    }

    let average = sum / count;

    let mut dev_sum = 0f32;

    for (_,row) in teachers.iter() {
        dev_sum += row.iter().fold(0.,|acc,x| acc + (x - average) * (x -average));
    }

    let dev_sta = (dev_sum / count).sqrt();

    let mut teachers = teachers.iter().map(|(t,row)| {
        (*t,row.iter().map(|&x| (x - average) / dev_sta).collect::<Vec<f32>>())
    }).collect::<Vec<(usize,Vec<f32>)>>();

    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    for _ in 0..2 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f32,6>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[t] = 1f32;

            let lossf = CrossEntropyMulticlass::new();

            net.train(expected, input, &mut optimizer, &lossf).unwrap();
        }
    }

    let mut tests:Vec<(usize,Vec<f32>)> = Vec::new();

    let mut reader =  Reader::from_path(
        Path::new("data")
                .join("penguins")
                .join("testing")
                .join("penguins_lter.csv")).unwrap();

    let headers = reader.headers().unwrap().iter().map(|c| c.trim().to_string()).collect::<Vec<String>>();

    for row in reader.records() {
        let columns = headers.iter().zip(row.unwrap().iter()).map(|(f,c)| (f.to_string(),c.to_string())).collect::<Vec<(String,String)>>();

        if columns.len() != 17 {
            continue;
        }

        let s = columns[2].1.split(' ').map(|c| c.to_string()).collect::<Vec<String>>();

        let t = species.get(&s[0] as &str).unwrap();

        let columns = columns.iter()
            .filter(|(f,_)| targets.contains(f.as_str().trim()))
            .map(|(_,c)| c.trim().to_owned())
            .filter(|c| !c.parse::<f32>().is_err())
            .map(|c| c.parse::<f32>().unwrap())
            .collect::<Vec<f32>>();
        if columns.len() < 6 {
            continue;
        }
        tests.push((*t,columns));
    }

    let count = tests.len() as f32;

    let mut sum = 0f32;

    for (_,row) in tests.iter() {
        sum += row.iter().fold(0.,|acc,x| acc + x);
    }

    let average = sum / count;

    let mut dev_sum = 0f32;

    for (_,row) in tests.iter() {
        dev_sum += row.iter().fold(0.,|acc,x| acc + (x - average) * (x -average));
    }

    let dev_sta = (dev_sum / count).sqrt();

    let tests = tests.iter().map(|(t,row)| {
        (*t,row.iter().map(|&x| (x - average) / dev_sta).collect::<Vec<f32>>())
    }).collect::<Vec<(usize,Vec<f32>)>>();

    let mut correct_answers = 0;

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f32, 6>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        if r == t {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {}",correct_answers);
    let rate = correct_answers as f32 * 100f32 / tests.len() as f32;
    print!("rate = {}",rate);
    debug_assert!(rate > 26.);
}
#[test]
fn test_penguins_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/6f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/50f32.sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,6>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,6,50>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,50,3>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut targets = HashSet::new();

    targets.insert("Culmen Length (mm)");
    targets.insert("Culmen Depth (mm)");
    targets.insert("Flipper Length (mm)");
    targets.insert("Body Mass (g)");
    targets.insert("Delta 15 N (o/oo)");
    targets.insert("Delta 13 C (o/oo)");

    let mut species:HashMap<&str,usize> = HashMap::new();

    species.insert("Adelie",0);
    species.insert("Chinstrap",1);
    species.insert("Gentoo",2);

    let mut teachers:Vec<(usize,Vec<f32>)> = Vec::new();

    let mut reader =  Reader::from_path(
        Path::new("data")
            .join("penguins")
            .join("training")
            .join("penguins_lter.csv")).unwrap();

    let headers = reader.headers().unwrap().iter().map(|c| c.trim().to_string()).collect::<Vec<String>>();

    for row in reader.records() {
        let columns= headers.iter().zip(row.unwrap().iter()).map(|(f,c)| (f.to_string(),c.to_string())).collect::<Vec<(String,String)>>();
        if columns.len() != 17 {
            continue;
        }

        let s = columns[2].1.split(' ').map(|s| s.to_string()).collect::<Vec<String>>();

        let t = species.get(&s[0] as &str).unwrap();

        let columns = columns.iter()
            .filter(|(f,_)| targets.contains(f.as_str().trim()))
            .map(|(_,c)| c.trim().to_owned())
            .filter(|c| !c.parse::<f32>().is_err())
            .map(|c| c.parse::<f32>().unwrap())
            .collect::<Vec<f32>>();
        if columns.len() < 6 {
            continue;
        }
        teachers.push((*t,columns));
    }

    let count = teachers.len() as f32;
    let mut sum = 0f32;

    for (_,row) in teachers.iter() {
        sum += row.iter().fold(0.,|acc,x| acc + x);
    }

    let average = sum / count;

    let mut dev_sum = 0f32;

    for (_,row) in teachers.iter() {
        dev_sum += row.iter().fold(0.,|acc,x| acc + (x - average) * (x -average));
    }

    let dev_sta = (dev_sum / count).sqrt();

    let mut teachers = teachers.iter().map(|(t,row)| {
        (*t,row.iter().map(|&x| (x - average) / dev_sta).collect::<Vec<f32>>())
    }).collect::<Vec<(usize,Vec<f32>)>>();

    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    for _ in 0..2 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f32,6>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[t] = 1f32;

            let lossf = CrossEntropyMulticlass::new();

            net.train(expected, input, &mut optimizer, &lossf).unwrap();
        }
    }

    let mut tests:Vec<(usize,Vec<f32>)> = Vec::new();

    let mut reader =  Reader::from_path(
        Path::new("data")
            .join("penguins")
            .join("testing")
            .join("penguins_lter.csv")).unwrap();

    let headers = reader.headers().unwrap().iter().map(|c| c.trim().to_string()).collect::<Vec<String>>();

    for row in reader.records() {
        let columns = headers.iter().zip(row.unwrap().iter()).map(|(f,c)| (f.to_string(),c.to_string())).collect::<Vec<(String,String)>>();

        if columns.len() != 17 {
            continue;
        }

        let s = columns[2].1.split(' ').map(|c| c.to_string()).collect::<Vec<String>>();

        let t = species.get(&s[0] as &str).unwrap();

        let columns = columns.iter()
            .filter(|(f,_)| targets.contains(f.as_str().trim()))
            .map(|(_,c)| c.trim().to_owned())
            .filter(|c| !c.parse::<f32>().is_err())
            .map(|c| c.parse::<f32>().unwrap())
            .collect::<Vec<f32>>();
        if columns.len() < 6 {
            continue;
        }
        tests.push((*t,columns));
    }

    let count = tests.len() as f32;

    let mut sum = 0f32;

    for (_,row) in tests.iter() {
        sum += row.iter().fold(0.,|acc,x| acc + x);
    }

    let average = sum / count;

    let mut dev_sum = 0f32;

    for (_,row) in tests.iter() {
        dev_sum += row.iter().fold(0.,|acc,x| acc + (x - average) * (x -average));
    }

    let dev_sta = (dev_sum / count).sqrt();

    let tests = tests.iter().map(|(t,row)| {
        (*t,row.iter().map(|&x| (x - average) / dev_sta).collect::<Vec<f32>>())
    }).collect::<Vec<(usize,Vec<f32>)>>();

    let mut correct_answers = 0;

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f32, 6>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        if r == t {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {}",correct_answers);
    let rate = correct_answers as f32 * 100f32 / tests.len() as f32;
    print!("rate = {}",rate);
    debug_assert!(rate > 26.);
}

#[test]
fn test_weather_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceGpu::new(&SHARED_MEMORY_POOL.clone()).unwrap();

    let net:InputLayer<f32,Arr<f32,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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

    let mut optimizer = MomentumSGD::new(0.001);

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

            let lossf = CrossEntropy::new();

            net.train(expected, input, &mut optimizer, &lossf).unwrap();
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

        let r = net.forward_all(input).unwrap();

        println!("晴れの確率 {}%",r[0]);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            println!("正解!");
            correct_answers += 1;
        } else {
            println!("不正解...");
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}
#[test]
fn test_weather_by_forward_diff_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/14f32).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

    let device = DeviceGpu::new(&SHARED_MEMORY_POOL.clone()).unwrap();

    let net:InputLayer<f32,DiffInput<DiffArr<f32,14>,f32,14,100>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        DiffLinearLayer::<_,_,_,DeviceGpu<f32>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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

    let mut optimizer = MomentumSGD::new(0.001);

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

            let lossf = CrossEntropy::new();

            net.train(expected, DiffInput::NotDiff(input), &mut optimizer, &lossf).unwrap();
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
                    acc.push(i,d).unwrap();
                    acc
                });

            prev = input.clone();

            let o = net.ask_diff_input(&s);

            Some(net.forward_diff(DiffInput::Diff(d,o)).unwrap())
        } else {
            prev = input.clone();

            Some(net.forward_diff(DiffInput::NotDiff(input)).unwrap())
        };

        let r = s.as_ref().map(|r| r.1[0]).unwrap();

        if (t && r >= 0.5) || !t && r < 0.5 {
            correct_answers += 1;
        }
    }

    println!("rate = {}",correct_answers as f32 / tests.len() as f32 * 100.);
    debug_assert!(correct_answers as f32 / tests.len() as f32 * 100. >= 73.);
}
#[test]
fn test_penguins_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/6f64).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, 1f64/50f64.sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f64,Arr<f64,6>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,6,50>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,50,3>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut targets = HashSet::new();

    targets.insert("Culmen Length (mm)");
    targets.insert("Culmen Depth (mm)");
    targets.insert("Flipper Length (mm)");
    targets.insert("Body Mass (g)");
    targets.insert("Delta 15 N (o/oo)");
    targets.insert("Delta 13 C (o/oo)");

    let mut species:HashMap<&str,usize> = HashMap::new();

    species.insert("Adelie",0);
    species.insert("Chinstrap",1);
    species.insert("Gentoo",2);

    let mut teachers:Vec<(usize,Vec<f64>)> = Vec::new();

    let mut reader =  Reader::from_path(
        Path::new("data")
            .join("penguins")
            .join("training")
            .join("penguins_lter.csv")).unwrap();

    let headers = reader.headers().unwrap().iter().map(|c| c.trim().to_string()).collect::<Vec<String>>();

    for row in reader.records() {
        let columns= headers.iter().zip(row.unwrap().iter()).map(|(f,c)| (f.to_string(),c.to_string())).collect::<Vec<(String,String)>>();
        if columns.len() != 17 {
            continue;
        }

        let s = columns[2].1.split(' ').map(|s| s.to_string()).collect::<Vec<String>>();

        let t = species.get(&s[0] as &str).unwrap();

        let columns = columns.iter()
            .filter(|(f,_)| targets.contains(f.as_str().trim()))
            .map(|(_,c)| c.trim().to_owned())
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        if columns.len() < 6 {
            continue;
        }
        teachers.push((*t,columns));
    }

    let count = teachers.len() as f64;
    let mut sum = 0f64;

    for (_,row) in teachers.iter() {
        sum += row.iter().fold(0.,|acc,x| acc + x);
    }

    let average = sum / count;

    let mut dev_sum = 0f64;

    for (_,row) in teachers.iter() {
        dev_sum += row.iter().fold(0.,|acc,x| acc + (x - average) * (x -average));
    }

    let dev_sta = (dev_sum / count).sqrt();

    let mut teachers = teachers.iter().map(|(t,row)| {
        (*t,row.iter().map(|&x| (x - average) / dev_sta).collect::<Vec<f64>>())
    }).collect::<Vec<(usize,Vec<f64>)>>();

    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    for _ in 0..2 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f64,6>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[t] = 1f64;

            let lossf = CrossEntropyMulticlass::new();

            net.train(expected, input, &mut optimizer, &lossf).unwrap();
        }
    }

    let mut tests:Vec<(usize,Vec<f64>)> = Vec::new();

    let mut reader =  Reader::from_path(
        Path::new("data")
            .join("penguins")
            .join("testing")
            .join("penguins_lter.csv")).unwrap();

    let headers = reader.headers().unwrap().iter().map(|c| c.trim().to_string()).collect::<Vec<String>>();

    for row in reader.records() {
        let columns = headers.iter().zip(row.unwrap().iter()).map(|(f,c)| (f.to_string(),c.to_string())).collect::<Vec<(String,String)>>();

        if columns.len() != 17 {
            continue;
        }

        let s = columns[2].1.split(' ').map(|c| c.to_string()).collect::<Vec<String>>();

        let t = species.get(&s[0] as &str).unwrap();

        let columns = columns.iter()
            .filter(|(f,_)| targets.contains(f.as_str().trim()))
            .map(|(_,c)| c.trim().to_owned())
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        if columns.len() < 6 {
            continue;
        }
        tests.push((*t,columns));
    }

    let count = tests.len() as f64;

    let mut sum = 0f64;

    for (_,row) in tests.iter() {
        sum += row.iter().fold(0.,|acc,x| acc + x);
    }

    let average = sum / count;

    let mut dev_sum = 0f64;

    for (_,row) in tests.iter() {
        dev_sum += row.iter().fold(0.,|acc,x| acc + (x - average) * (x -average));
    }

    let dev_sta = (dev_sum / count).sqrt();

    let tests = tests.iter().map(|(t,row)| {
        (*t,row.iter().map(|&x| (x - average) / dev_sta).collect::<Vec<f64>>())
    }).collect::<Vec<(usize,Vec<f64>)>>();

    let mut correct_answers = 0;

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f64, 6>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        if r == t {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {}",correct_answers);
    let rate = correct_answers as f64 * 100f64 / tests.len() as f64;
    print!("rate = {}",rate);
    debug_assert!(rate > 26.);
}

#[test]
fn test_weather_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/14f64).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, 1f64/100f64.sqrt()).unwrap();

    let device = DeviceGpu::new(&SHARED_MEMORY_POOL.clone()).unwrap();

    let net:InputLayer<f64,Arr<f64,14>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut teachers:Vec<(bool,Vec<f64>)> = Vec::new();

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
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap() / 10000.)
            .collect::<Vec<f64>>();
        if columns.len() < 14 {
            continue;
        }

        teachers.push((t,columns));
    }

    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    teachers.shuffle(&mut rng);

    for _ in 0..1 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f64,14>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[0] = if t {
                1.
            } else {
                0.
            };

            let lossf = CrossEntropy::new();

            net.train(expected, input, &mut optimizer, &lossf).unwrap();
        }
    }

    let mut tests:Vec<(bool,Vec<f64>)> = Vec::new();

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
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap() / 10000.)
            .collect::<Vec<f64>>();
        if columns.len() < 14 {
            continue;
        }

        tests.push((t,columns));
    }

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f64, 14>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        let r = net.forward_all(input).unwrap();

        println!("晴れの確率 {}%",r[0]);

        if (t && r[0] >= 0.5) || !t && r[0] < 0.5 {
            println!("正解!");
            correct_answers += 1;
        } else {
            println!("不正解...");
        }
    }

    println!("rate = {}",correct_answers as f64 / tests.len() as f64 * 100.);
    debug_assert!(correct_answers as f64 / tests.len() as f64 * 100. >= 73.);
}
#[test]
fn test_weather_by_forward_diff_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/14f64).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, 1f64/100f64.sqrt()).unwrap();

    let device = DeviceGpu::new(&SHARED_MEMORY_POOL.clone()).unwrap();

    let net:InputLayer<f64,DiffInput<DiffArr<f64,14>,f64,14,100>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        DiffLinearLayer::<_,_,_,DeviceGpu<f64>,_,14,100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f64>,_,100,1>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut teachers:Vec<(bool,Vec<f64>)> = Vec::new();

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
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap() / 10000.)
            .collect::<Vec<f64>>();
        if columns.len() < 14 {
            continue;
        }

        teachers.push((t,columns));
    }

    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    let mut correct_answers = 0;

    teachers.shuffle(&mut rng);

    for _ in 0..1 {
        teachers.shuffle(&mut rng);

        for (t, columns) in teachers.iter() {
            let t = *t;

            let mut input = Arr::<f64,14>::new();

            for (it, p) in input.iter_mut().zip(columns.iter()) {
                *it = *p;
            }

            let mut expected = Arr::new();

            expected[0] = if t {
                1.
            } else {
                0.
            };

            let lossf = CrossEntropy::new();

            net.train(expected, DiffInput::NotDiff(input), &mut optimizer, &lossf).unwrap();
        }
    }

    let mut tests:Vec<(bool,Vec<f64>)> = Vec::new();

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
            .filter(|c| !c.parse::<f64>().is_err())
            .map(|c| c.parse::<f64>().unwrap() / 10000.)
            .collect::<Vec<f64>>();
        if columns.len() < 14 {
            continue;
        }

        tests.push((t,columns));
    }

    let mut s = None;
    let mut prev = Arr::new();

    for (t, columns) in tests.iter() {
        let t = *t;

        let mut input = Arr::<f64, 14>::new();

        for (it, p) in input.iter_mut().zip(columns.iter()) {
            *it = *p;
        }

        s = if let Some(s) = s {
            let d = input.iter().enumerate().zip(prev.iter())
                .filter(|((_,&input),&p)| input != p)
                .map(|((index,&input),&p)| (index,input - p))
                .fold(DiffArr::new(),| mut acc,(i,d) | {
                    acc.push(i,d).unwrap();
                    acc
                });

            prev = input.clone();

            let o = net.ask_diff_input(&s);

            Some(net.forward_diff(DiffInput::Diff(d,o)).unwrap())
        } else {
            prev = input.clone();

            Some(net.forward_diff(DiffInput::NotDiff(input)).unwrap())
        };

        let r = s.as_ref().map(|r| r.1[0]).unwrap();

        if (t && r >= 0.5) || !t && r < 0.5 {
            correct_answers += 1;
        }
    }

    println!("rate = {}",correct_answers as f64 / tests.len() as f64 * 100.);

    debug_assert!(correct_answers as f64 / tests.len() as f64 * 100. >= 73.);
}
#[test]
fn test_mnist_sigmoid_and_crossentropy() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(100f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();
        
                let n = *n;
        
                let mut input = Arr::<f32, 784>::new();
        
                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }
        
                let mut expected = Arr::new();

                expected[0] = if n % 2 == 0 {
                    1.0
                } else {
                    0.0
                };

                (expected, input)
            }).fold((Vec::<Arr<f32, 1>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropy::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.iter().len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
        }

        let r = net.forward_all(input).unwrap();

        println!("n = {}, r = {}",n,r[0]);

        if (n % 2 == 0 && r[0] >= 0.5) || (n % 2 == 1 && r[0] < 0.5){
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. >= 80.)
}
#[test]
fn test_mnist_sigmoid_and_crossentropy_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(100f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();
        
                let n = *n;
        
                let mut input = Arr::<f32, 784>::new();
        
                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }
        
                let mut expected = Arr::new();

                expected[0] = if n % 2 == 0 {
                    1.0
                } else {
                    0.0
                };

                (expected, input)
            }).fold((Vec::<Arr<f32, 1>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropy::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.iter().len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
        }

        let r = net.forward_all(input).unwrap();

        println!("n = {}, r = {}",n,r[0]);

        if (n % 2 == 0 && r[0] >= 0.5) || (n % 2 == 1 && r[0] < 0.5){
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. >= 80.)
}
#[test]
fn test_mnist_tanh_and_relu_and_mse() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(100f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Tanh::new(&device),&device)
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();
        
                let n = *n;
        
                let mut input = Arr::<f32, 784>::new();
        
                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }
        
                let mut expected = Arr::new();

                expected[0] = if n % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };

                (expected, input)
            }).fold((Vec::<Arr<f32, 1>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = Mse::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.iter().len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
        }

        let r = net.forward_all(input).unwrap();

        println!("n = {}, r = {}",n,r[0]);

        if (n % 2 == 0 && r[0] >= 0.0) || (n % 2 == 1 && r[0] < 0.0){
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. >= 80.)
}
#[test]
fn test_mnist_tanh_and_relu_and_mse_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(100f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Tanh::new(&device),&device)
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();
        
                let n = *n;
        
                let mut input = Arr::<f32, 784>::new();
        
                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }

                let mut expected = Arr::new();

                expected[0] = if n % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };

                (expected, input)
            }).fold((Vec::<Arr<f32, 1>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = Mse::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.iter().len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
        }

        let r = net.forward_all(input).unwrap();

        println!("n = {}, r = {}",n,r[0]);

        if (n % 2 == 0 && r[0] >= 0.0) || (n % 2 == 1 && r[0] < 0.0){
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. >= 80.)
}
#[test]
fn test_mnist_tanh_and_swish_and_mse() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(100f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Swish::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Swish::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceCpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
    }).add_layer(|l| {
        ActivationLayer::new(l,Tanh::new(&device),&device)
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();
        
                let n = *n;
        
                let mut input = Arr::<f32, 784>::new();
        
                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }
        
                let mut expected = Arr::new();

                expected[0] = if n % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };

                (expected, input)
            }).fold((Vec::<Arr<f32, 1>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = Mse::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.iter().len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
        }
        let r = net.forward_all(input).unwrap();

        println!("n = {}, r = {}",n,r[0]);

        if (n % 2 == 0 && r[0] >= 0.0) || (n % 2 == 1 && r[0] < 0.0){
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. >= 80.)
}
#[test]
fn test_mnist_tanh_and_swish_and_mse_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(100f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,{ 28*28 },100>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Swish::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Swish::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayer::<_,_,_,DeviceGpu<f32>,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Tanh::new(&device),&device)
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
    let mut optimizer = MomentumSGD::new(0.001);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(120) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes();
        
                let n = *n;
        
                let mut input = Arr::<f32, 784>::new();
        
                for (it, &p) in input.iter_mut().zip(pixels) {
                    *it = p as f32 / 255.;
                }
        
                let mut expected = Arr::new();

                expected[0] = if n % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };

                (expected, input)
            }).fold((Vec::<Arr<f32, 1>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = Mse::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
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

    let count = tests.iter().len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes();

        let n = *n;

        let mut input = Arr::<f32, 784>::new();

        for (it, &p) in input.iter_mut().zip(pixels) {
            *it = p as f32 / 255.;
        }

        let r = net.forward_all(input).unwrap();

        println!("n = {}, r = {}",n,r[0]);

        if (n % 2 == 0 && r[0] >= 0.0) || (n % 2 == 1 && r[0] < 0.0){
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. >= 80.);
}