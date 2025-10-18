use nncombinator::scheduler::{LambdaLR, LinearWarmupLR, Scheduler};

#[test]
fn test_scheduler_seq() {
    let mut scheduler = LinearWarmupLR::new(10, 0.01)
                                        .seq(10, LinearWarmupLR::new(20, 0.01))
                                        .seq(20,LambdaLR::new(0.01, |_| Ok(0.001)));

    assert_eq!(scheduler.schedule(0.01, 0).unwrap(), 0.0);
    assert_eq!(scheduler.schedule(0.01, 9).unwrap(), 0.01*(9.0/10.0));
    assert_eq!(scheduler.schedule(0.01, 10).unwrap(), 0.0);
    assert_eq!(scheduler.schedule(0.01, 11).unwrap(), 0.01/20.0);
    assert_eq!(scheduler.schedule(0.01, 29).unwrap(), 0.01*(19.0/20.0));
    assert_eq!(scheduler.schedule(0.01, 30).unwrap(), 0.00001);
    assert_eq!(scheduler.schedule(0.01, 35).unwrap(), 0.00001);
    assert_eq!(scheduler.schedule(0.01, 40).unwrap(), 0.00001);
    assert_eq!(scheduler.schedule(0.01, 50).unwrap(), 0.00001);
}