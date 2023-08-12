use std::cell::RefCell;
use std::fs;
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use mnist::{Mnist, MnistBuilder};
use rayon::prelude::{ParallelIterator};
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, SoftMax};
use nncombinator::arr::Arr;
use nncombinator::device::{DeviceCpu, DeviceGpu};
use nncombinator::layer::{ActivationLayer, AddLayer, AddLayerTrain, BatchForward, BatchTrain, ForwardAll, InputLayer, LinearLayerBuilder, LinearOutputLayer};
use nncombinator::layer::batchnormalization::{BatchNormalizationLayerBuilder};
use nncombinator::lossfunction::CrossEntropyMulticlass;
use nncombinator::optimizer::{MomentumSGD};

use crate::common::SHARED_MEMORY_POOL;

#[test]
fn test_mnist_batch_norm() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/300f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(300f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<{ 28*28 },300>().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    //}).add_layer(|l| {
    //    ActivationLayer::new(l,ReLu::new(&device),&device)
    //}).add_layer(|l| {
    //    let rnd = rnd.clone();
    //    LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    //}).add_layer(|l| {
    //    BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    //}).add_layer(|l| {
    //    ActivationLayer::new(l,ReLu::new(&device),&device)
    //}).add_layer(|l| {
    //    let rnd = rnd.clone();
    //    LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    //}).add_layer(|l| {
    //    BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,10>().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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
    let mut optimizer = MomentumSGD::new(0.08);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 1000;
    let batch_size = 100;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(usize,PathBuf)>>();

    let iter_per_epoch = (train_size / batch_size).max(1);
    let max_epochs = 10;
    let mut epoch_cnt = 0;

    for i in 0.. {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for _ in 0..1 {
            let batch_data = teachers.iter().take(batch_size).map(|(n, path)| {
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

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        if i % iter_per_epoch == 0 {
            epoch_cnt += 1;
            println!("total_loss = {}", total_loss);
            println!("loss_average = {}", total_loss as f32 / count as f32);

            if epoch_cnt >= max_epochs {
                break;
            }
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
    let n2 = Normal::<f32>::new(0.0, (2f32/300f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(300f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<{ 28*28 },300>().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,10>().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
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

    let mut optimizer = MomentumSGD::new(0.08);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 2000;
    let batch_size = 20;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(Vec<f32>,usize)>>();

    let iter_per_epoch = (train_size / batch_size).max(1);
    let max_epochs = 10;
    let mut epoch_cnt = 0;

    for i in 0.. {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for _ in 0..1 {
            let batch_data = teachers.iter().take(batch_size).map(|( img,lbl)| {
                let pixels = img.clone().try_into().unwrap();

                let input = pixels;

                let mut expected = Arr::new();

                expected[*lbl] = 1.;

                (expected, input)
            }).fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        if i % iter_per_epoch == 0 {
            epoch_cnt += 1;
            println!("total_loss = {}", total_loss);
            println!("loss_average = {}", total_loss as f32 / count as f32);

            if epoch_cnt >= max_epochs {
                break;
            }
        }
    }

    let mut tests:Vec<(Vec<f32>,usize)> = Vec::new();

    for (i,&l) in tst_img.chunks(28*28).zip(tst_lbl.iter()) {
        tests.push((i.iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>(),l as usize));
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(100);

    for (img,lbl) in tests.iter().take(100) {
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

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 75.)
}
#[test]
fn test_mnist_batch_norm_for_gpu() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/300f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(300f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<{ 28*28 },300>().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
        //}).add_layer(|l| {
        //    ActivationLayer::new(l,ReLu::new(&device),&device)
        //}).add_layer(|l| {
        //    let rnd = rnd.clone();
        //    LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
        //}).add_layer(|l| {
        //    BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
        //}).add_layer(|l| {
        //    ActivationLayer::new(l,ReLu::new(&device),&device)
        //}).add_layer(|l| {
        //    let rnd = rnd.clone();
        //    LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
        //}).add_layer(|l| {
        //    BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,10>().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
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
    let mut optimizer = MomentumSGD::new(0.08);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 1000;
    let batch_size = 100;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(usize,PathBuf)>>();

    let iter_per_epoch = (train_size / batch_size).max(1);
    let max_epochs = 10;
    let mut epoch_cnt = 0;

    for i in 0.. {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for _ in 0..1 {
            let batch_data = teachers.iter().take(batch_size).map(|(n, path)| {
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

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        if i % iter_per_epoch == 0 {
            epoch_cnt += 1;
            println!("total_loss = {}", total_loss);
            println!("loss_average = {}", total_loss as f32 / count as f32);

            if epoch_cnt >= max_epochs {
                break;
            }
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
    let n2 = Normal::<f32>::new(0.0, (2f32/300f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(300f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<{ 28*28 },300>().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,300>().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<300,10>().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
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

    let mut optimizer = MomentumSGD::new(0.08);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let train_size = 2000;
    let batch_size = 20;

    let mut teachers = teachers.into_iter().take(train_size).collect::<Vec<(Vec<f32>,usize)>>();

    let iter_per_epoch = (train_size / batch_size).max(1);
    let max_epochs = 10;
    let mut epoch_cnt = 0;

    for i in 0.. {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for _ in 0..1 {
            let batch_data = teachers.iter().take(batch_size).map(|( img,lbl)| {
                let pixels = img.clone().try_into().unwrap();

                let input = pixels;

                let mut expected = Arr::new();

                expected[*lbl] = 1.;

                (expected, input)
            }).fold((Vec::<Arr<f32, 10>>::new(), Vec::<Arr<f32, 784>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i);
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            count += 1;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }

        if i % iter_per_epoch == 0 {
            epoch_cnt += 1;
            println!("total_loss = {}", total_loss);
            println!("loss_average = {}", total_loss as f32 / count as f32);

            if epoch_cnt >= max_epochs {
                break;
            }
        }
    }

    let mut tests:Vec<(Vec<f32>,usize)> = Vec::new();

    for (i,&l) in tst_img.chunks(28*28).zip(tst_lbl.iter()) {
        tests.push((i.iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>(),l as usize));
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(100);

    for (img,lbl) in tests.iter().take(100) {
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

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 75.)
}
