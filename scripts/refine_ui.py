from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'app/page.tsx'
s=p.read_text(encoding='utf-8-sig').replace("[view,setView]=useState('orbit')", "[view,setView]=useState('pack')")
s=s.replace('<div className="view-buttons">', '<div className="view-buttons"><button aria-pressed={view===\'pack\'} onClick={()=>setView(\'pack\')}>馬群</button>')
p.write_text(s,encoding='utf-8')
p=root/'app/TrackScene.tsx'
s=p.read_text(encoding='utf-8').replace("scene.add(new T.HemisphereLight('#fff6e4','#839c75',2.6))", "scene.add(new T.HemisphereLight('#fff6e4','#839c75',1.8))").replace("DirectionalLight('#fff7e8',3)", "DirectionalLight('#fff7e8',1.8)")
s=s.replace('controls.update();renderer.render(scene,camera);', "if(view==='pack'){const center=horses.reduce((v,g)=>v.add(g.position),new T.Vector3()).multiplyScalar(1/horses.length);controls.target.lerp(center,.12);camera.position.lerp(center.clone().add(new T.Vector3(12,22,30)),.10);} controls.update();renderer.render(scene,camera);")
p.write_text(s,encoding='utf-8')
