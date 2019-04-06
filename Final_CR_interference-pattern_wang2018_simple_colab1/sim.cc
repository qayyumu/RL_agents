/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 */

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/packet-sink.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/stats-module.h"
#include "ns3/flow-monitor-module.h"

#include "mygym.h"
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <stdlib.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Interference-Pattern");


int main (int argc, char *argv[])
{
  // Parameters of the environment
  uint32_t simSeed = 1;
  double simulationTime = 10; //seconds
  double envStepTime = 0.1; //seconds, ns3gym env step time interval
  uint32_t openGymPort = 5555;
  uint32_t testArg = 0;

  //Parameters of the scenario
  uint32_t nodeNum = 2;
  double distance = 10.0;
  bool enableFading = false;
  double interfererPower = 10;

  // Interference Pattern
  double interferenceSlotTime = 0.1;  // seconds;

  //        time,    channel usage
  
  /*
  std::map<uint32_t, std::vector<uint32_t> > interferencePattern;
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (0, {1,0,0,0}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (1, {0,1,0,0}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (2, {0,0,1,0}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (3, {0,0,0,1}));
*/

  std::map<uint32_t, std::vector<uint32_t> > interferencePattern;
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (0,  {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (1,  {1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (2,  {1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (3,  {1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (4,  {1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (5,  {1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (6,  {1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (7,  {1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (8,  {1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (9,  {1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (10, {1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (11, {1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (12, {1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (13, {1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (14, {1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1}));
  interferencePattern.insert(std::pair<uint32_t, std::vector<uint32_t> > (15, {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0}));
  
  
  // set channel number correctly, do not modify
  std::vector<uint32_t> tmp = interferencePattern.at(0);
  uint32_t channNum = tmp.size();

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
  cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.AddValue ("enableFading", "If fading should be enabled. Default: false", enableFading);
  cmd.Parse (argc, argv);

  NS_LOG_UNCOND("Ns3Env parameters:");
  NS_LOG_UNCOND("--simulationTime: " << simulationTime);
  NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND("--seed: " << simSeed);
  NS_LOG_UNCOND("--testArg: " << testArg);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);

  // OpenGym Env
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
  Ptr<MyGymEnv> myGymEnv = CreateObject<MyGymEnv> (channNum);
  myGymEnv->SetOpenGymInterface(openGymInterface);

  NodeContainer nodes;
  nodes.Create (nodeNum);

  // Channel
  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  Ptr<NakagamiPropagationLossModel> fadingModel = CreateObject<NakagamiPropagationLossModel> ();
  if (enableFading) {
    lossModel->SetNext (fadingModel);
  }
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  // Mobility model
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (0.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (distance),
                                 "DeltaY", DoubleValue (distance),
                                 "GridWidth", UintegerValue (nodeNum),  // will create linear topology
                                 "LayoutType", StringValue ("RowFirst"));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);

  // Define channel models
  std::map<uint32_t, Ptr<SpectrumModel> > spectrumModels;
  for (uint32_t chanId=0; chanId<channNum; chanId++) {
    double fc = 5200e6 + 20e6 * chanId;
    BandInfo bandInfo;
    bandInfo.fc = fc;
    bandInfo.fl = fc - 10e6;
    bandInfo.fh = fc + 10e6;
    Bands bands;
    bands.push_back (bandInfo);
    Ptr<SpectrumModel> sm = Create<SpectrumModel> (bands);
    spectrumModels.insert(std::pair<uint32_t, Ptr<SpectrumModel>>(chanId, sm));
  }

  // Spectrum Analyzer --- Channel Sensing
  Ptr<Node> sensingNode = nodes.Get(0);
  SpectrumAnalyzerHelper spectrumAnalyzerHelper;
  spectrumAnalyzerHelper.SetChannel (spectrumChannel);
  spectrumAnalyzerHelper.SetPhyAttribute ("Resolution", TimeValue (MilliSeconds (100)));
  spectrumAnalyzerHelper.SetPhyAttribute ("NoisePowerSpectralDensity", DoubleValue (1e-15));     // -120 dBm/Hz
  NetDeviceContainer spectrumAnalyzers;
  
  for (uint32_t chanId=0; chanId<channNum; chanId++) {
    spectrumAnalyzerHelper.SetRxSpectrumModel (spectrumModels.at(chanId));
    spectrumAnalyzers.Add(spectrumAnalyzerHelper.Install (sensingNode));
  }

  for (uint32_t i=0; i< spectrumAnalyzers.GetN(); i++)
  {
    Ptr<NetDevice> netDev = spectrumAnalyzers.Get(i);
    Ptr<NonCommunicatingNetDevice> nonCommNetDev = DynamicCast<NonCommunicatingNetDevice>(netDev);
    Ptr<Object> spectrumPhy = nonCommNetDev->GetPhy();
    Ptr<SpectrumAnalyzer> spectrumAnalyzer = DynamicCast<SpectrumAnalyzer>(spectrumPhy);
    spectrumAnalyzer->Start ();

    std::ostringstream oss;
    oss.str ("");
    uint32_t devId = netDev->GetIfIndex();
    oss << "/NodeList/" << sensingNode->GetId () << "/DeviceList/" << devId << "/$ns3::NonCommunicatingNetDevice/Phy/AveragePowerSpectralDensityReport";
    uint32_t channelId = i;
    Config::ConnectWithoutContext (oss.str (),MakeBoundCallback (&MyGymEnv::PerformCca, myGymEnv, channelId));
  }

  // Signal Generator --- Generate interference pattern
  Ptr<Node> interferingNode = nodes.Get(1);
  WaveformGeneratorHelper waveformGeneratorHelper;
  waveformGeneratorHelper.SetChannel (spectrumChannel);
  waveformGeneratorHelper.SetPhyAttribute ("Period", TimeValue (Seconds (interferenceSlotTime)));
  waveformGeneratorHelper.SetPhyAttribute ("DutyCycle", DoubleValue (1.0));
  NetDeviceContainer waveformGeneratorDevices;

  for (uint32_t chanId=0; chanId<channNum; chanId++) {
    Ptr<SpectrumValue> wgPsd = Create<SpectrumValue> (spectrumModels.at(chanId));
    *wgPsd = interfererPower / (20e6);
    NS_LOG_DEBUG ("wgPsd : " << *wgPsd << " integrated power: " << Integral (*(GetPointer (wgPsd))));
    waveformGeneratorHelper.SetTxPowerSpectralDensity (wgPsd);
    waveformGeneratorDevices.Add(waveformGeneratorHelper.Install (interferingNode));
  }


boost::mt19937 gen;
  // Schedule interference pattern
double scheduledTime = 0;
int oct_sav[16] = {0};
  /*
    we are modelling a markov process:
    state 1: emit {1, 0, 0, 0}
    state 2: emit {...
    T
    0.7, 0.1, 0.1, 0.1
    0.1, 0.7, 0.1, 0.1
    0.1, 0.1, 0.7, 0.1
    0.1, 0.1, 0.1, 0.7
  */
    
 /* 
  std::vector<boost::random::discrete_distribution<> > T{
                    boost::random::discrete_distribution<> ({0.7, 0.1, 0.1, 0.1}),
                    boost::random::discrete_distribution<> ({0.1, 0.7, 0.1, 0.1}),
                    boost::random::discrete_distribution<> ({0.1, 0.1, 0.7, 0.1}),
                    boost::random::discrete_distribution<> ({0.1, 0.1, 0.1, 0.7})};
   */
   
   std::vector<boost::random::discrete_distribution<> > T{
                    boost::random::discrete_distribution<> ({0.7, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.7, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.7, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.7,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.7, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.7, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.7, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.7,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.7, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.7, 0.02, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.7, 0.02,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.7,0.02, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.7, 0.02, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.7, 0.02, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.7, 0.02}),
                    boost::random::discrete_distribution<> ({0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.7}),
                  
     
     
   };
   
   
   
 if (1)
 {
  // Schedule interference pattern
  double scheduledTime = 0;
  
  while (scheduledTime < simulationTime)
  {
    std::map<uint32_t, std::vector<uint32_t> >::iterator it;
    for (it=interferencePattern.begin(); it!=interferencePattern.end(); it++) {
      std::vector<uint32_t> channels = (*it).second;

      for (uint32_t chanId = 0; chanId < channels.size(); chanId++){
        uint32_t occupied = channels.at(chanId);
        NS_LOG_DEBUG("scheduledTime: " << scheduledTime << " ChanId " << chanId << " Occupied: " << occupied);
        //NS_LOG_UNCOND("scheduledTime: " << scheduledTime << " ChanId " << chanId << " Occupied: " << occupied);
        //NS_LOG_UNCOND(occupied);
        oct_sav[chanId]= occupied;
        if (occupied == 1) {
          Simulator::Schedule (Seconds (scheduledTime), &WaveformGenerator::Start,
                               waveformGeneratorDevices.Get (chanId)->GetObject<NonCommunicatingNetDevice> ()->GetPhy ()->GetObject<WaveformGenerator> ());
        } else {
         // Simulator::Schedule (Seconds (scheduledTime), &WaveformGenerator::Stop,
           //                    waveformGeneratorDevices.Get (chanId)->GetObject<NonCommunicatingNetDevice> ()->GetPhy ()->GetObject<WaveformGenerator> ());
        
          Simulator::Schedule (Seconds (fmax( 0.0, scheduledTime-0.0001)), &WaveformGenerator::Stop,waveformGeneratorDevices.Get (chanId)->GetObject<NonCommunicatingNetDevice> ()->GetPhy ()->GetObject<WaveformGenerator> ());
 
          
        }
      }
     // NS_LOG_UNCOND(oct_sav[0] << "," << oct_sav[1] << "," << oct_sav[2] << "," << oct_sav[3] << "," << oct_sav[4] << "," << oct_sav[5] << "," << oct_sav[6] << "," << oct_sav[7] << "," << oct_sav[8] << "," << oct_sav[9] << "," << oct_sav[10] << "," << oct_sav[11] << "," << oct_sav[12] << "," << oct_sav[13]<< "," << oct_sav[14]<< "," << oct_sav[15]);
      scheduledTime += interferenceSlotTime;
    }
  }
 }
 else
 {
  
  //double curstate = 0; //rand() % 4;
  double curstate = rand() % 16;
  while (scheduledTime < simulationTime)
  {
    std::vector<uint32_t> channels = interferencePattern[curstate];
     //NS_LOG_UNCOND("\n");
    for (uint32_t chanId = 0; chanId < channels.size(); chanId++){
      uint32_t occupied = channels.at(chanId);
      NS_LOG_DEBUG("scheduledTime: " << scheduledTime << " ChanId " << chanId << " Occupied: " << occupied);
      //NS_LOG_UNCOND("scheduledTime: " << scheduledTime << " ChanId " << chanId << " Occupied: " << occupied);
      //NS_LOG_UNCOND("scheduledTime: " << scheduledTime << " ChanId " << chanId << " Occupied: " << occupied);
      oct_sav[chanId]= occupied;
      if (occupied == 1) {
        Simulator::Schedule (Seconds (scheduledTime), &WaveformGenerator::Start,
                             waveformGeneratorDevices.Get (chanId)->GetObject<NonCommunicatingNetDevice> ()->GetPhy ()->GetObject<WaveformGenerator> ());
      } else {
      //  Simulator::Schedule (Seconds (scheduledTime), &WaveformGenerator::Stop,
        //                     waveformGeneratorDevices.Get (chanId)->GetObject<NonCommunicatingNetDevice> ()->GetPhy ()->GetObject<WaveformGenerator> ());
        Simulator::Schedule (Seconds (fmax( 0.0, scheduledTime-0.0001)), &WaveformGenerator::Stop,waveformGeneratorDevices.Get (chanId)->GetObject<NonCommunicatingNetDevice> ()->GetPhy ()->GetObject<WaveformGenerator> ());
 
        
        
      }
    }
    
   
       // fmax( 0.0, scheduledTime-0.0001)) 
  
    //NS_LOG_UNCOND(oct_sav[0] << "," << oct_sav[1] << "," << oct_sav[2] << "," << oct_sav[3]);
    //NS_LOG_UNCOND(oct_sav[0] << "," << oct_sav[1] << "," << oct_sav[2] << "," << oct_sav[3] << "," << oct_sav[4] << "," << oct_sav[5] << "," << oct_sav[6] << "," << oct_sav[7] << "," << oct_sav[8] << "," << oct_sav[9] << "," << oct_sav[10] << "," << oct_sav[11] << "," << oct_sav[12] << "," << oct_sav[13]<< "," << oct_sav[14]<< "," << oct_sav[15]);
     NS_LOG_UNCOND(oct_sav[0] << "," << oct_sav[1] << "," << oct_sav[2] << "," << oct_sav[3] << "," << oct_sav[4] << "," << oct_sav[5] << "," << oct_sav[6] << "," << oct_sav[7] << "," << oct_sav[8] << "," << oct_sav[9] << "," << oct_sav[10] << "," << oct_sav[11] << "," << oct_sav[12] << "," << oct_sav[13]<< "," << oct_sav[14]<< "," << oct_sav[15]);

    scheduledTime += interferenceSlotTime;
    curstate = T[curstate](gen);
    
  }

   
 }

  NS_LOG_UNCOND ("Simulation start");
  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  NS_LOG_UNCOND ("Simulation stop");
  myGymEnv->NotifySimulationEnd();

  Simulator::Destroy ();
  NS_LOG_UNCOND ("Simulation exit");
}
