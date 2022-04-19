use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;
use crate::error::*;

pub trait Persistence<U,P> {
    fn load(&mut self, persistence:&mut P) -> Result<(),ConfigReadError>;
    fn save(&mut self, persistence:&mut P);
}
pub trait SaveToFile<U> {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error>;
}
pub trait ReadFromPersistence<U> {
    fn read(&mut self) -> U;
}
pub trait SaveToPersistence<U> {
    fn write(&mut self);
}
pub trait LinearPersistence<U> {
    fn read(&mut self) -> Result<U, ConfigReadError>;
    fn write(&mut self,u:U);
    fn verify_eof(&mut self) -> Result<(),ConfigReadError>;
}
pub enum UnitOrMarker<U> {
    Unit(U),
    LayerStart,
    UnitsStart
}
pub struct TextFilePersistence<U> where U: FromStr + Sized {
    reader:BufReader<File>,
    line:Option<Vec<String>>,
    index:usize,
    data:Vec<UnitOrMarker<U>>
}
impl<U> TextFilePersistence<U> where U: FromStr + Sized {
    pub fn new (file:&str) -> Result<TextFilePersistence<U>,ConfigReadError> {
        Ok(TextFilePersistence {
            reader:BufReader::new(OpenOptions::new().read(true).create(false).open(file)?),
            line:None,
            index:0usize,
            data:Vec::new()
        })
    }

    fn read_line(&mut self) -> Result<String, ConfigReadError> {
        let mut buf = String::new();
        let n = self.reader.read_line(&mut buf)?;

        buf = buf.trim().to_string();

        if n == 0 {
            Err(ConfigReadError::InvalidState(String::from(
                "End of input has been reached.")))
        } else {
            Ok(buf)
        }
    }

    fn next_token(&mut self) -> Result<String, ConfigReadError> {
        let t = match self.line {
            None => {
                self.index = 0;
                let mut buf = self.read_line()?;

                while match &*buf {
                    "" => true,
                    s => match s.chars().nth(0) {
                        Some('#') => true,
                        _ => false,
                    }
                } {
                    buf = self.read_line()?;
                }

                let line = buf.split(" ").map(|s| s.to_string()).collect::<Vec<String>>();
                let t = (&line[self.index]).clone();
                self.line = Some(line);
                t
            },
            Some(ref line) => {
                (&line[self.index]).clone()
            }
        };

        self.index = self.index + 1;

        if match self.line {
            Some(ref line) if self.index >= line.len() => {
                true
            },
            Some(_) => {
                false
            }
            None => false,
        } {
            self.line = None;
        }

        Ok(t)
    }

    pub fn verify_eof(&mut self) -> Result<(),ConfigReadError> {
        let mut buf = String::new();

        loop {
            let n = self.reader.read_line(&mut buf)?;

            if n == 0 {
                return Ok(());
            }

            buf = buf.trim().to_string();

            if !buf.is_empty() {
                return Err(ConfigReadError::InvalidState(
                    String::from("Data loaded , but the input has not reached the end.")));
            } else {
                buf.clear();
            }
        }
    }
}
impl<U> TextFilePersistence<U> where U: FromStr + Sized, ConfigReadError: From<<U as FromStr>::Err> {
    pub fn read(&mut self) -> Result<U, ConfigReadError> {
        Ok(self.next_token()?.parse::<U>()?)
    }
}
impl<U> TextFilePersistence<U> where U: FromStr + Sized {
    pub fn write(&mut self,v:UnitOrMarker<U>) {
        self.data.push(v);
    }
}
impl<U> SaveToFile<U> for TextFilePersistence<U> where U: FromStr + Sized + Display {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error> {
        let mut bw = BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?);

        for u in self.data.iter() {
            match u {
                UnitOrMarker::Unit(u) => {
                    bw.write(format!("{} ",u).as_bytes())?;
                },
                UnitOrMarker::LayerStart => {
                    bw.write(b"#layer\n")?;
                },
                UnitOrMarker::UnitsStart => {
                    bw.write(b"\n")?;
                }
            }
        }

        Ok(())
    }
}
pub struct BinFilePersistence<U> {
    reader:BufReader<File>,
    data:Vec<U>
}
impl<U> BinFilePersistence<U> {
    pub fn new (file:&str) -> Result<BinFilePersistence<U>, ConfigReadError> {
        Ok(BinFilePersistence {
            reader:BufReader::new(OpenOptions::new().read(true).create(false).open(file)?),
            data:Vec::new()
        })
    }
}
impl LinearPersistence<f64> for BinFilePersistence<f64> {
    fn read(&mut self) -> Result<f64, ConfigReadError> {
        let mut buf = [0; 8];

        self.reader.read_exact(&mut buf)?;

        Ok(f64::from_bits(
            (buf[0] as u64) << 56 |
                (buf[1] as u64) << 48 |
                (buf[2] as u64) << 40 |
                (buf[3] as u64) << 32 |
                (buf[4] as u64) << 24 |
                (buf[5] as u64) << 16 |
                (buf[6] as u64) << 8  |
                buf[7] as u64)
        )
    }

    fn write(&mut self, u: f64) {
        self.data.push(u);
    }

    fn verify_eof(&mut self) -> Result<(), ConfigReadError> {
        let mut buf:[u8; 1] = [0];

        let n = self.reader.read(&mut buf)?;

        if n == 0 {
            Ok(())
        } else {
            Err(ConfigReadError::InvalidState(String::from("Data loaded , but the input has not reached the end.")))
        }
    }
}
impl LinearPersistence<f32> for BinFilePersistence<f32> {
    fn read(&mut self) -> Result<f32, ConfigReadError> {
        let mut buf = [0; 4];

        self.reader.read_exact(&mut buf)?;

        Ok(f32::from_bits(
                (buf[0] as u32) << 24 |
                (buf[1] as u32) << 16 |
                (buf[2] as u32) << 8  |
                buf[3] as u32)
        )
    }

    fn write(&mut self, u: f32) {
        self.data.push(u);
    }

    fn verify_eof(&mut self) -> Result<(), ConfigReadError> {
        let mut buf:[u8; 1] = [0];

        let n = self.reader.read(&mut buf)?;

        if n == 0 {
            Ok(())
        } else {
            Err(ConfigReadError::InvalidState(String::from("Data loaded , but the input has not reached the end.")))
        }
    }
}
impl SaveToFile<f64> for BinFilePersistence<f64> {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error> {
        let mut bw = BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?);

        for u in self.data.iter() {
            let mut buf = [0; 8];
            let bits = u.to_bits();

            buf[0] = (bits >> 56 & 0xff) as u8;
            buf[1] = (bits >> 48 & 0xff) as u8;
            buf[2] = (bits >> 40 & 0xff) as u8;
            buf[3] = (bits >> 32 & 0xff) as u8;
            buf[4] = (bits >> 24 & 0xff) as u8;
            buf[5] = (bits >> 16 & 0xff) as u8;
            buf[6] = (bits >> 8 & 0xff) as u8;
            buf[7] = (bits & 0xff) as u8;

            bw.write(&buf)?;
        }

        Ok(())
    }
}
impl SaveToFile<f32> for BinFilePersistence<f32> {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error> {
        let mut bw = BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?);

        for u in self.data.iter() {
            let mut buf = [0; 4];
            let bits = u.to_bits();
            buf[0] = (bits >> 24 & 0xff) as u8;
            buf[1] = (bits >> 16 & 0xff) as u8;
            buf[2] = (bits >> 8 & 0xff) as u8;
            buf[3] = (bits & 0xff) as u8;

            bw.write(&buf)?;
        }

        Ok(())
    }
}
