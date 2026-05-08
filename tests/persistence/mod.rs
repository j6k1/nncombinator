
use std::cell::RefCell;
use std::{fs};
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Sigmoid, SoftMax};
use nncombinator::arr::{Arr};
use nncombinator::device::{DeviceGpu};
use nncombinator::layer::{AddLayer, BatchForward, BatchTrain, ForwardAll, PersistProgress, Step};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::batchnormalization::BatchNormalizationLayerBuilder;
use nncombinator::layer::bias::BiasLayerBuilder;
use nncombinator::layer::bridge::{BridgeLayerBuilder};
use nncombinator::layer::input::{InputLayer};
use nncombinator::layer::linear::{LinearLayerBuilder};
use nncombinator::layer::logging::LoggingLayer;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::lossfunction::{CrossEntropyMulticlass};
use nncombinator::optimizer::{AdamBuilder};
use nncombinator::persistence::{BinFilePersistence, Persistence, SaveToFile, TextFilePersistence};
use crate::common::{assert_text_persist_progress, SHARED_MEMORY_POOL};

#[test]
fn test_mnist_for_gpu_with_persistence() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/512f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(256f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },512>::new().build(l,&device,
                                                         move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                         &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<512,256>::new().build(l,&device,
                                                   move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<256,10>::new().build(l,&device,
                                                  move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
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

    let mut teachers = teachers.into_iter().take(60000).collect::<Vec<(usize,PathBuf)>>();

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

    let nn_path = Path::new("data").join("tmp").join("mnist_for_gpu_with_persistence.bin");

    let mut p = BinFilePersistence::new(&nn_path).unwrap();

    net.save(&mut p).unwrap();

    p.save(&nn_path).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },512>::new().build(l,&device,
                                                         move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                         &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<512,256>::new().build(l,&device,
                                                   move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<256,10>::new().build(l,&device,
                                                  move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
    });

    let nn_path = Path::new("data").join("tmp").join("mnist_for_gpu_with_persistence.bin");

    let mut p = BinFilePersistence::new(&nn_path).unwrap();

    net.load(&mut p).unwrap();

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

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 90.)
}
#[test]
fn test_mnist_for_gpu_with_text_persistence() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/512f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(256f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },512>::new().build(l,&device,
                                                         move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                         &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<512,256>::new().build(l,&device,
                                                   move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<256,10>::new().build(l,&device,
                                                  move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
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

    let mut teachers = teachers.into_iter().take(60000).collect::<Vec<(usize,PathBuf)>>();

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

    let nn_path = Path::new("data").join("tmp").join("mnist_for_gpu_with_persistence.txt");

    let mut p = TextFilePersistence::new(&nn_path).unwrap();

    net.save(&mut p).unwrap();

    p.save(&nn_path).unwrap();

    let net:InputLayer<f32,Arr<f32,{ 28*28 }>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<{ 28*28 },512>::new().build(l,&device,
                                                         move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                         &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<512,256>::new().build(l,&device,
                                                   move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                   &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<256,10>::new().build(l,&device,
                                                  move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
    });

    let nn_path = Path::new("data").join("tmp").join("mnist_for_gpu_with_persistence.txt");

    let mut p = TextFilePersistence::new(&nn_path).unwrap();

    net.load(&mut p).unwrap();

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

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 90.)
}
#[test]
fn test_gpu_with_persist_progress_all_layears() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(32f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/32f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(32f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,32>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32,32>::new().build(l,&device,
                                                         move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                         &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 32>::new().build(l, &device,
                                                  move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BiasLayerBuilder::new().build(l,&device,|| 0., &optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                 move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BridgeLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer(|l| {
        LoggingLayer::new(l,&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
    });

    net.step().unwrap();

    let nn_path = Path::new("data").join("tmp").join("test_gpu_with_persist_progress_all_layears.bin");

    let mut p = BinFilePersistence::new(&nn_path).unwrap();

    net.save_progress(&mut p).unwrap();

    p.save(&nn_path).unwrap();

    let net:InputLayer<f32,Arr<f32,32>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32,32>::new().build(l,&device,
                                                 move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 32>::new().build(l, &device,
                                                  move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BiasLayerBuilder::new().build(l,&device,|| 0., &optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                 move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BridgeLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer(|l| {
        LoggingLayer::new(l,&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
    });

    let nn_path = Path::new("data").join("tmp").join("test_gpu_with_persist_progress_all_layears.bin");

    let mut p = BinFilePersistence::new(&nn_path).unwrap();

    net.load_progress(&mut p).unwrap();

    assert!(true)
}
#[test]
fn test_gpu_with_persist_to_text_progress_all_layears() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(32f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/32f32).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, 1f32/(32f32).sqrt()).unwrap();

    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let device = DeviceGpu::new(memory_pool).unwrap();

    let net:InputLayer<f32,Arr<f32,32>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        assert_text_persist_progress(&l);

        let rnd = rnd.clone();
        LinearLayerBuilder::<32,32>::new().build(l,&device,
                                                 move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 32>::new().build(l, &device,
                                                  move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        BiasLayerBuilder::new().build(l,&device,|| 0., &optimizer_builder).unwrap()
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                 move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        BridgeLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        LoggingLayer::new(l,&device)
    }).add_layer(|l| {
        assert_text_persist_progress(&l);

        LinearOutputLayer::new(l,&device).unwrap()
    });

    assert_text_persist_progress(&net);

    net.step().unwrap();

    let nn_path = Path::new("data").join("tmp").join("test_gpu_with_persist_progress_all_layears.txt");

    let mut p = TextFilePersistence::new(&nn_path).unwrap();

    net.save_progress(&mut p).unwrap();

    p.save(&nn_path).unwrap();

    let net:InputLayer<f32,Arr<f32,32>,_,_> = InputLayer::new(&device);

    let rnd = rnd_base.clone();

    let optimizer_builder = AdamBuilder::new(&device).lr(0.005).weight_decay(0.0001);

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32,32>::new().build(l,&device,
                                                 move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BatchNormalizationLayerBuilder::new().build(l,&device,&optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 32>::new().build(l, &device,
                                                  move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                  &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BiasLayerBuilder::new().build(l,&device,|| 0., &optimizer_builder).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::<32, 1>::new().build(l, &device,
                                                 move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.,
                                                 &optimizer_builder
        ).unwrap()
    }).add_layer(|l| {
        BridgeLayerBuilder::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,Sigmoid::new(&device),&device)
    }).add_layer(|l| {
        LoggingLayer::new(l,&device)
    }).add_layer(|l| {
        LinearOutputLayer::new(l,&device).unwrap()
    });

    let nn_path = Path::new("data").join("tmp").join("test_gpu_with_persist_progress_all_layears.txt");

    let mut p = TextFilePersistence::new(&nn_path).unwrap();

    net.load_progress(&mut p).unwrap();

    assert!(true)
}
