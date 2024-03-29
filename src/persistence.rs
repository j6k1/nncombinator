//! Implementation on persistence of neural network models

use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;
use crate::error::*;

pub trait Persistence<U,P,K> where K: PersistenceType {
    /// Load Model
    /// # Arguments
    /// * `persistence` - model persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    fn load(&mut self, persistence:&mut P) -> Result<(),ConfigReadError>;
    /// Save Model
    /// # Arguments
    /// * `persistence` - model persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`PersistenceError`]
    fn save(&mut self, persistence:&mut P) -> Result<(), PersistenceError>;
}
pub trait PersistenceType {}
pub struct Specialized;
pub struct Linear;
impl PersistenceType for Specialized {}
impl PersistenceType for Linear {}

/// Trait that defines the implementation of the ability to save a model to a file
pub trait SaveToFile<U> {
    /// Save to File
    /// # Arguments
    /// * `file` - Destination path
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`io::Error`]
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error>;
}
/// Trait to define an implementation to persist the model in a flat data structure
pub trait LinearPersistence<U> {
    /// Read to restore the persisted model
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    fn read(&mut self) -> Result<U, ConfigReadError>;
    /// Write to persist model information
    /// # Arguments
    /// * `u` - Weight value
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`PersistenceError`]
    fn write(&mut self, u:U) -> Result<(), PersistenceError>;
    /// Has the read position of the persisted information reached EOF?
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    fn verify_eof(&mut self) -> Result<(),ConfigReadError>;
}
/// Types for passing identifiable information about layers and unit boundaries when persisting models
pub enum UnitOrMarker<U> {
    /// Not a boundary.
    Unit(U),
    /// layer boundary
    LayerStart,
    /// boundary
    UnitsStart
}
/// Persistent object for saving to a text file
pub struct TextFilePersistence<U> where U: FromStr + Sized {
    reader:Option<BufReader<File>>,
    line:Option<Vec<String>>,
    index:usize,
    data:Vec<UnitOrMarker<U>>
}
impl<U> TextFilePersistence<U> where U: FromStr + Sized {
    /// Create an instance of TextFilePersistence
    /// # Arguments
    /// * `file` - File path to be persisted
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    pub fn new<P: AsRef<Path>>(file:P) -> Result<TextFilePersistence<U>,ConfigReadError> {
        if file.as_ref().exists() {
            Ok(TextFilePersistence {
                reader:Some(BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)),
                line: None,
                index: 0usize,
                data: Vec::new()
            })
        } else {
            Ok(TextFilePersistence {
                reader:None,
                line: None,
                index: 0usize,
                data: Vec::new()
            })
        }
    }

    fn read_line(&mut self) -> Result<String, ConfigReadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = String::new();
                let n = reader.read_line(&mut buf)?;

                buf = buf.trim().to_string();

                if n == 0 {
                    Err(ConfigReadError::InvalidState(String::from(
                        "End of input has been reached.")))
                } else {
                    Ok(buf)
                }
            },
            None => {
                Err(ConfigReadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
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

    /// Has the read position of the persisted information reached EOF?
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    pub fn verify_eof(&mut self) -> Result<(),ConfigReadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = String::new();

                loop {
                    let n = reader.read_line(&mut buf)?;

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
            },
            None => {
                Err(ConfigReadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }
}
impl<U> TextFilePersistence<U> where U: FromStr + Sized, ConfigReadError: From<<U as FromStr>::Err> {
    /// Read the weight values from a file
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    pub fn read(&mut self) -> Result<U, ConfigReadError> {
        Ok(self.next_token()?.parse::<U>()?)
    }
}
impl<U> TextFilePersistence<U> where U: FromStr + Sized {
    /// Layer weights are added to the end of the internal buffer
    /// # Arguments
    /// * `v` - Weight value
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
/// Trait that defines a Persistence implementation
/// that stores and loads in fixed length record format.
pub struct BinFilePersistence<U> {
    reader:Option<BufReader<File>>,
    data:Vec<U>
}
impl<U> BinFilePersistence<U> {
    /// Create an instance of TextFilePersistence
    /// # Arguments
    /// * `file` - File path to be persisted
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ConfigReadError`]
    pub fn new<P: AsRef<Path>>(file:P) -> Result<BinFilePersistence<U>, ConfigReadError> {
        if file.as_ref().exists() {
            Ok(BinFilePersistence {
                reader:Some(BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)),
                data:Vec::new()
            })
        } else {
            Ok(BinFilePersistence {
                reader:None,
                data:Vec::new()
            })
        }
    }
}
impl LinearPersistence<f64> for BinFilePersistence<f64> {
    fn read(&mut self) -> Result<f64, ConfigReadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = [0; 8];

                reader.read_exact(&mut buf)?;

                Ok(f64::from_bits(
                    (buf[0] as u64) << 56 |
                        (buf[1] as u64) << 48 |
                        (buf[2] as u64) << 40 |
                        (buf[3] as u64) << 32 |
                        (buf[4] as u64) << 24 |
                        (buf[5] as u64) << 16 |
                        (buf[6] as u64) << 8 |
                        buf[7] as u64)
                )
            },
            None => {
                Err(ConfigReadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }

    fn write(&mut self, u: f64) -> Result<(), PersistenceError> {
        self.data.push(u);
        Ok(())
    }

    fn verify_eof(&mut self) -> Result<(), ConfigReadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf: [u8; 1] = [0];

                let n = reader.read(&mut buf)?;

                if n == 0 {
                    Ok(())
                } else {
                    Err(ConfigReadError::InvalidState(String::from("Data loaded , but the input has not reached the end.")))
                }
            },
            None => {
                Err(ConfigReadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }
}
impl LinearPersistence<f32> for BinFilePersistence<f32> {
    fn read(&mut self) -> Result<f32, ConfigReadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = [0; 4];

                reader.read_exact(&mut buf)?;

                Ok(f32::from_bits(
                    (buf[0] as u32) << 24 |
                        (buf[1] as u32) << 16 |
                        (buf[2] as u32) << 8 |
                        buf[3] as u32)
                )
            },
            None => {
                Err(ConfigReadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }

    fn write(&mut self, u: f32) -> Result<(), PersistenceError> {
        self.data.push(u);
        Ok(())
    }

    fn verify_eof(&mut self) -> Result<(), ConfigReadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf: [u8; 1] = [0];

                let n = reader.read(&mut buf)?;

                if n == 0 {
                    Ok(())
                } else {
                    Err(ConfigReadError::InvalidState(String::from("Data loaded , but the input has not reached the end.")))
                }
            },
            None => {
                Err(ConfigReadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
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
