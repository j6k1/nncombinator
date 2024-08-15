use std::cell::RefCell;
use std::fs;
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use mnist::{Mnist, MnistBuilder};
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, SoftMax};
use nncombinator::arr::Arr;
use nncombinator::device::{DeviceCpu, DeviceGpu};
use nncombinator::layer::{AddLayer, AddLayerTrain, BatchForward, BatchTrain, ForwardAll};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::batchnormalization::{BatchNormalizationLayerBuilder};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::linear::LinearLayerBuilder;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::lossfunction::CrossEntropyMulticlass;
use nncombinator::optimizer::{MomentumSGDBuilder};
use crate::common::SHARED_MEMORY_POOL;

#[test]
fn test_mnist_batch_norm() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder =  MomentumSGDBuilder::new(&device).lr(0.004);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },100>::new().build(l,&device,
            move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,100>::new().build(l,&device,
            move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,10>::new().build(l,&device,
            move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
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

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 40000;
    let batch_size = 200;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..5 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
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

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
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
fn test_fashion_mnist_batch_norm() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/2000f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(1800f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.01);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },2000>::new().build(l,&device,
                                                          move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                          &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<2000,2000>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<2000,1800>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<1800,10>::new().build(l,&device,
                                                   move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path(Path::new("mnist").join("fashion").to_str().unwrap())
        .label_format_digit()
        .use_fashion_data()
        .finalize();

    let mut teachers:Vec<(Vec<f32>,usize)> = Vec::new();

    for (i,&l) in trn_img.chunks(28*28).zip(trn_lbl.iter()) {
        teachers.push((i.iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>(),l as usize));
    }

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 6000;
    let batch_size = 256;

    let teachers = teachers.into_iter().take(train_size).collect::<Vec<(Vec<f32>,usize)>>();

    let mut teachers = teachers.into_iter().map(|(img, lbl)| {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let mut expected = Arr::new();

        expected[lbl] = 1.;

        (expected, input)
    }).collect::<Vec<(Arr<f32,10>,Arr<f32,784>)>>();

    let max_epochs = 10;

    for _ in 0..max_epochs {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
            let batch_data = teachers.iter().cloned().fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
            total_loss += loss;

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
    }

    let mut tests:Vec<(Vec<f32>,usize)> = Vec::new();

    for (i,&l) in tst_img.chunks(28*28).zip(tst_lbl.iter()) {
        tests.push((i.iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>(),l as usize));
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(1000);

    for (img,lbl) in tests.iter().take(1000) {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        let n = *lbl;

        if n == r {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 80.)
}
#[test]
fn test_mnist_batch_norm_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/(28f64*28f64)).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, (2f64/100f64).sqrt()).unwrap();
    let n3 = Normal::<f64>::new(0.0, 1f64/(100f64).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f64,Arr<f64,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.004);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },100>::new().build(l,&device,
            move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,100>::new().build(l,&device,
            move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,10>::new().build(l,&device,
            move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
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

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 40000;
    let batch_size = 200;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..5 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
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

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
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

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f64 / count as f64 * 100.);

    debug_assert!(correct_answers as f64 / count as f64 * 100. > 80.)
}
#[test]
fn test_fashion_mnist_batch_norm_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/(28f64*28f64)).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, (2f64/2000f64).sqrt()).unwrap();
    let n3 = Normal::<f64>::new(0.0, 1f64/(1800f64).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f64,Arr<f64,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.01);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },2000>::new().build(l,&device,
                                                          move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                          &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<2000,2000>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<2000,1800>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<1800,10>::new().build(l,&device,
                                                   move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path(Path::new("mnist").join("fashion").to_str().unwrap())
        .label_format_digit()
        .use_fashion_data()
        .finalize();

    let mut teachers:Vec<(Vec<f64>,usize)> = Vec::new();

    for (i,&l) in trn_img.chunks(28*28).zip(trn_lbl.iter()) {
        teachers.push((i.iter().map(|&p| p as f64 / 255.).collect::<Vec<f64>>(),l as usize));
    }

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 6000;
    let batch_size = 256;

    let teachers = teachers.into_iter().take(train_size).collect::<Vec<(Vec<f64>,usize)>>();

    let mut teachers = teachers.into_iter().map(|(img, lbl)| {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let mut expected = Arr::new();

        expected[lbl] = 1.;

        (expected, input)
    }).collect::<Vec<(Arr<f64,10>,Arr<f64,784>)>>();

    let max_epochs = 10;

    for _ in 0..max_epochs {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
            let batch_data = teachers.iter().cloned().fold((Vec::<Arr<f64, 10>>::new(), Vec::<Arr<f64, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
            total_loss += loss;

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f64 / count as f64);
    }

    let mut tests:Vec<(Vec<f64>,usize)> = Vec::new();

    for (i,&l) in tst_img.chunks(28*28).zip(tst_lbl.iter()) {
        tests.push((i.iter().map(|&p| p as f64 / 255.).collect::<Vec<f64>>(),l as usize));
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(1000);

    for (img,lbl) in tests.iter().take(1000) {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        let n = *lbl;

        if n == r {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 80.)
}
#[test]
fn test_mnist_batch_norm_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/100f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(100f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.004);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },100>::new().build(l,&device,
            move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,100>::new().build(l,&device,
            move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,10>::new().build(l,&device,
            move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
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

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 60000;
    let batch_size = 120;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
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

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
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
fn test_fashion_mnist_batch_norm_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/2000f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(1800f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.01);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },1000>::new().build(l,&device,
            move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<1000,1000>::new().build(l,&device,
            move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<1000,1000>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<1000,10>::new().build(l,&device,
            move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path(Path::new("mnist").join("fashion").to_str().unwrap())
        .label_format_digit()
        .use_fashion_data()
        .finalize();

    let mut teachers:Vec<(Vec<f32>,usize)> = Vec::new();

    for (i,&l) in trn_img.chunks(28*28).zip(trn_lbl.iter()) {
        teachers.push((i.iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>(),l as usize));
    }

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 6000;
    let batch_size = 256;

    let teachers = teachers.into_iter().take(train_size).collect::<Vec<(Vec<f32>,usize)>>();

    let mut teachers = teachers.into_iter().map(|(img, lbl)| {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let mut expected = Arr::new();

        expected[lbl] = 1.;

        (expected, input)
    }).collect::<Vec<(Arr<f32,10>,Arr<f32,784>)>>();

    let max_epochs = 10;

    for _ in 0..max_epochs {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
            let batch_data = teachers.iter().cloned().fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
            total_loss += loss;

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss / count as f32);
    }

    let mut tests:Vec<(Vec<f32>,usize)> = Vec::new();

    for (i,&l) in tst_img.chunks(28*28).zip(tst_lbl.iter()) {
        tests.push((i.iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>(),l as usize));
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(1000);

    for (img,lbl) in tests.iter().take(1000) {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        let n = *lbl;

        if n == r {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 80.)
}
#[test]
fn test_mnist_batch_norm_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/(28f64*28f64)).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, (2f64/100f64).sqrt()).unwrap();
    let n3 = Normal::<f64>::new(0.0, 1f64/(100f64).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f64,Arr<f64,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.004);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },100>::new().build(l,&device,
            move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,100>::new().build(l,&device,
            move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<100,10>::new().build(l,&device,
            move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
            &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
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

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 60000;
    let batch_size = 120;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..3 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
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

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss / count as f64);
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
fn test_fashion_mnist_batch_norm_for_gpu_double() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f64>::new(0.0, (2f64/(28f64*28f64)).sqrt()).unwrap();
    let n2 = Normal::<f64>::new(0.0, (2f64/2000f64).sqrt()).unwrap();
    let n3 = Normal::<f64>::new(0.0, 1f64/(1800f64).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f64,Arr<f64,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = MomentumSGDBuilder::new(&device).lr(0.01);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },2000>::new().build(l,&device,
                                                          move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                          &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<2000,2000>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<2000,1800>::new().build(l,&device,
                                                     move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                     &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<1800,10>::new().build(l,&device,
                                                   move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path(Path::new("mnist").join("fashion").to_str().unwrap())
        .label_format_digit()
        .use_fashion_data()
        .finalize();

    let mut teachers:Vec<(Vec<f64>,usize)> = Vec::new();

    for (i,&l) in trn_img.chunks(28*28).zip(trn_lbl.iter()) {
        teachers.push((i.iter().map(|&p| p as f64 / 255.).collect::<Vec<f64>>(),l as usize));
    }

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 6000;
    let batch_size = 256;

    let teachers = teachers.into_iter().take(train_size).collect::<Vec<(Vec<f64>,usize)>>();

    let mut teachers = teachers.into_iter().map(|(img, lbl)| {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let mut expected = Arr::new();

        expected[lbl] = 1.;

        (expected, input)
    }).collect::<Vec<(Arr<f64,10>,Arr<f64,784>)>>();

    let max_epochs = 10;

    for _ in 0..max_epochs {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(batch_size) {
            let batch_data = teachers.iter().cloned().fold((Vec::<Arr<f64, 10>>::new(), Vec::<Arr<f64, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &lossf).unwrap();
            total_loss += loss;

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss / count as f64);
    }

    let mut tests:Vec<(Vec<f64>,usize)> = Vec::new();

    for (i,&l) in tst_img.chunks(28*28).zip(tst_lbl.iter()) {
        tests.push((i.iter().map(|&p| p as f64 / 255.).collect::<Vec<f64>>(),l as usize));
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(1000);

    for (img,lbl) in tests.iter().take(1000) {
        let pixels = img.clone().try_into().unwrap();

        let input = pixels;

        let r = net.forward_all(input).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        let n = *lbl;

        if n == r {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f64 / count as f64 * 100.);

    debug_assert!(correct_answers as f64 / count as f64 * 100. > 80.)
}
