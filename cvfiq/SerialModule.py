import serial
import time
import logging
import serial.tools.list_ports

class SerialObject:
    """
    Allow to transmit data to a Serial Device like Arduino.
    Example send $255255000
    """
    def __init__(self, portNo=None, baudRate=9600, digits=1, timeout=1):
        """
        Initialize the serial object.
        :param portNo: Port Number.
        :param baudRate: Baud Rate.
        :param digits: Number of digits per value to send
        :param timeout: Read timeout in seconds
        """
        self.portNo = portNo
        self.baudRate = baudRate
        self.digits = digits
        self.timeout = timeout
        self.ser = None
        connected = False
        if self.portNo is None:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if "Arduino" in p.description:
                    print(f'{p.description} Connected')
                    self.ser = serial.Serial(p.device, baudRate, timeout=self.timeout)
                    connected = True
            if not connected:
                logging.warning("Arduino Not Found. Please enter COM Port Number instead.")

        else:
            try:
                self.ser = serial.Serial(self.portNo, self.baudRate, timeout=self.timeout)
                print("Serial Device Connected")
            except serial.SerialException as e:
                logging.warning(f"Serial Device Not Connected: {e}")


    def sendData(self, data):
        """
        Send data to the Serial device.
        :param data: list of values to send
        :return: True if sent successfully, False otherwise
        """
        if self.ser is None:
            return False
        myString = "$"
        for d in data:
            myString += str(int(d)).zfill(self.digits)
        try:
            self.ser.write(myString.encode())
            return True
        except serial.SerialException:
            return False

    def getData(self, timeout=None):
        """
        Receive a line of data from the Serial device.
        :param timeout: Override read timeout in seconds (None uses instance default)
        :return: list of data values, or empty list on timeout/error
        """
        if self.ser is None:
            return []
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            data = self.ser.readline()
            if not data:
                return []
            data = data.decode("utf-8").strip()
            return [d for d in data.split('#') if d]
        except serial.SerialException:
            return []

def main():
    arduino = SerialObject()
    while True:
        arduino.sendData([1, 1, 1, 1, 1])
        time.sleep(2)
        arduino.sendData([0, 0, 0, 0, 0])
        time.sleep(2)


if __name__ == "__main__":
    main()
